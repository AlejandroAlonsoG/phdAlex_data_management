"""Timing experiments for Gemini Flash API calls.

Benchmarks different thinking configurations on both Gemini SDKs to find the
fastest setup for the data_ordering pipeline.

SDKs tested:
1. google-genai (New SDK) — used by the current GeminiClient
2. google-generativeai (Old SDK) — the previous implementation

Each configuration is run SEQUENTIALLY (not in parallel) so that timing is
not skewed by server-side concurrency throttling.  A warmup call is made
first to prime the connection.

Usage:
    python -m data_ordering.test_llm_timing
    python -m data_ordering.test_llm_timing --model gemini-2.0-flash
    python -m data_ordering.test_llm_timing --runs 3
    python -m data_ordering.test_llm_timing --only-new   # skip old SDK
    python -m data_ordering.test_llm_timing --only-old   # skip new SDK
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

# Suppress deprecation warnings for old SDK
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load .env
try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / '.env'
    load_dotenv(_env if _env.exists() else None)
except ImportError:
    pass

# ─── SDK Imports ─────────────────────────────────────────────────────────────

# New SDK: google-genai
try:
    from google import genai as new_genai
    from google.genai import types as new_types
    NEW_SDK_AVAILABLE = True
except ImportError:
    new_genai = None
    new_types = None
    NEW_SDK_AVAILABLE = False

# Old SDK: google-generativeai
try:
    import google.generativeai as old_genai
    OLD_SDK_AVAILABLE = True
    try:
        from google.generativeai.types import GenerationConfig as OldGenerationConfig
    except ImportError:
        OldGenerationConfig = None
    try:
        from google.generativeai.types import ThinkingConfig as OldThinkingConfig
    except ImportError:
        OldThinkingConfig = None
except ImportError:
    old_genai = None
    OLD_SDK_AVAILABLE = False
    OldGenerationConfig = None
    OldThinkingConfig = None

# ─── Test Prompt (same one the pipeline uses, simplified) ────────────────────

SYSTEM_PROMPT = """You are an expert paleontologist analyzing directory paths from a fossil image database.
Your task: Extract taxonomic and collection information ONLY from the directory path structure.

Return JSON with fields: macroclass, taxonomic_class, genus, campaign_year, specimen_id, collection_code, confidence"""

SAMPLE_PATH = "/data/Las_Hoyas/2019/Artropodos/Insectos/Coleoptera/Delclosia"
USER_PROMPT = f"Analyze this directory path from a fossil image database:\n\nPath: {SAMPLE_PATH}\n\nExtract any taxonomic, collection, or campaign information visible in the path structure."

# JSON schema (mirrors PATH_ANALYSIS_SCHEMA from llm_integration.py)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "macroclass": {"type": "string", "nullable": True},
        "taxonomic_class": {"type": "string", "nullable": True},
        "genus": {"type": "string", "nullable": True},
        "campaign_year": {"type": "integer", "nullable": True},
        "specimen_id": {"type": "string", "nullable": True},
        "collection_code": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["macroclass", "taxonomic_class", "genus", "campaign_year",
                  "specimen_id", "collection_code", "confidence"],
}

# ─── Test Configurations ─────────────────────────────────────────────────────

# Each entry: (label, thinking_config_kwargs, use_json_schema)
# thinking_config_kwargs is None => no thinking config (server dynamic default)
#
# NOTE on model families:
#   - Gemini 3.x   → use thinkingLevel: MINIMAL, LOW, MEDIUM, HIGH
#   - Gemini 2.5.x → use thinkingBudget: 0 (off), -1 (dynamic), or 128-24576
#   - Gemini 2.0.x → deprecated, but thinkingLevel seems accepted
#   There is NO "NONE" level. MINIMAL is the lowest (Gemini 3 Flash only).

# --- thinkingLevel configs (Gemini 3 / 2.0-flash) ---
THINKING_LEVEL_CONFIGS = [
    ("No thinking config (default)",        None,                              False),
    ("Thinking: MINIMAL",                   {"thinking_level": "MINIMAL"},     False),
    ("Thinking: LOW",                       {"thinking_level": "LOW"},         False),
    ("Thinking: MEDIUM",                    {"thinking_level": "MEDIUM"},      False),
    ("Thinking: HIGH",                      {"thinking_level": "HIGH"},        False),
    # With structured JSON output (what the pipeline actually does)
    ("JSON + No thinking config",           None,                              True),
    ("JSON + Thinking: MINIMAL",            {"thinking_level": "MINIMAL"},     True),
    ("JSON + Thinking: LOW",                {"thinking_level": "LOW"},         True),
    ("JSON + Thinking: MEDIUM",             {"thinking_level": "MEDIUM"},      True),
]

# --- thinkingBudget configs (Gemini 2.5 models) ---
THINKING_BUDGET_CONFIGS = [
    ("No thinking config (default)",        None,                              False),
    ("Budget: 0 (thinking OFF)",            {"thinking_budget": 0},            False),
    ("Budget: 1024 (light)",                {"thinking_budget": 1024},         False),
    ("Budget: 4096 (moderate)",             {"thinking_budget": 4096},         False),
    ("Budget: -1 (dynamic)",                {"thinking_budget": -1},           False),
    # With structured JSON output
    ("JSON + Budget: 0 (OFF)",              {"thinking_budget": 0},            True),
    ("JSON + Budget: 1024 (light)",         {"thinking_budget": 1024},         True),
    ("JSON + Budget: -1 (dynamic)",         {"thinking_budget": -1},           True),
]


def select_configs(model: str):
    """Pick the right config set based on model family."""
    model_lower = model.lower()
    if '2.5' in model_lower or '25' in model_lower:
        print(f"  Model '{model}' detected as Gemini 2.5 → using thinkingBudget configs")
        return THINKING_BUDGET_CONFIGS
    else:
        # Gemini 3.x, 2.0, or unknown → use thinkingLevel
        print(f"  Model '{model}' → using thinkingLevel configs")
        return THINKING_LEVEL_CONFIGS


# ─── New SDK Runner ──────────────────────────────────────────────────────────

def run_new_sdk(model, api_key, label, thinking_kwargs, use_json):
    """Single test run on the new google-genai SDK."""
    client = new_genai.Client(api_key=api_key)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"

    config_params = {
        "temperature": 0.1,
        "max_output_tokens": 2048,
    }

    if use_json:
        config_params["response_mime_type"] = "application/json"
        config_params["response_json_schema"] = JSON_SCHEMA

    if thinking_kwargs:
        # thinking_kwargs can be {"thinking_level": "LOW"} or {"thinking_budget": 1024}
        config_params["thinking_config"] = new_types.ThinkingConfig(**thinking_kwargs)

    gen_config = new_types.GenerateContentConfig(**config_params)

    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=gen_config,
        )
        elapsed = time.perf_counter() - t0
        text = resp.text or ""

        # Extract token usage if available
        usage = {}
        if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
            um = resp.usage_metadata
            usage = {
                'prompt_tokens': getattr(um, 'prompt_token_count', None),
                'output_tokens': getattr(um, 'candidates_token_count', None),
                'thinking_tokens': getattr(um, 'thoughts_token_count', None),
                'total_tokens': getattr(um, 'total_token_count', None),
            }

        return {
            'sdk': 'google-genai (new)',
            'label': label,
            'ok': True,
            'elapsed': elapsed,
            'output_len': len(text),
            'usage': usage,
            'snippet': text[:200],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'sdk': 'google-genai (new)',
            'label': label,
            'ok': False,
            'elapsed': elapsed,
            'error': f"{type(e).__name__}: {e}",
        }


# ─── Old SDK Runner ──────────────────────────────────────────────────────────

def run_old_sdk(model, api_key, label, thinking_kwargs, use_json):
    """Single test run on the old google-generativeai SDK."""
    if not OLD_SDK_AVAILABLE:
        return {
            'sdk': 'google-generativeai (old)',
            'label': label,
            'ok': False,
            'elapsed': 0.0,
            'error': 'Package not installed (pip install google-generativeai)',
        }

    old_genai.configure(api_key=api_key)
    gm = old_genai.GenerativeModel(model)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"

    gen_kwargs = {
        "temperature": 0.1,
        "max_output_tokens": 2048,
    }

    if use_json:
        gen_kwargs["response_mime_type"] = "application/json"
        # Old SDK uses response_schema (not response_json_schema)
        gen_kwargs["response_schema"] = JSON_SCHEMA

    if thinking_kwargs:
        if OldThinkingConfig is None:
            return {
                'sdk': 'google-generativeai (old)',
                'label': label,
                'ok': False,
                'elapsed': 0.0,
                'error': 'ThinkingConfig not available in installed version',
            }
        # Old SDK ThinkingConfig may use lowercase level strings
        # Try as-is first; some versions accept either case
        try:
            gen_kwargs["thinking_config"] = OldThinkingConfig(**thinking_kwargs)
        except Exception as e1:
            # Try lowercase
            lower_kwargs = {}
            for k, v in thinking_kwargs.items():
                lower_kwargs[k] = v.lower() if isinstance(v, str) else v
            try:
                gen_kwargs["thinking_config"] = OldThinkingConfig(**lower_kwargs)
            except Exception:
                return {
                    'sdk': 'google-generativeai (old)',
                    'label': label,
                    'ok': False,
                    'elapsed': 0.0,
                    'error': f'ThinkingConfig rejected kwargs: {e1}',
                }

    if OldGenerationConfig is None:
        return {
            'sdk': 'google-generativeai (old)',
            'label': label,
            'ok': False,
            'elapsed': 0.0,
            'error': 'GenerationConfig not available in installed version',
        }

    try:
        gen_config = OldGenerationConfig(**gen_kwargs)
    except TypeError as e:
        return {
            'sdk': 'google-generativeai (old)',
            'label': label,
            'ok': False,
            'elapsed': 0.0,
            'error': f'GenerationConfig rejected args: {e}',
        }

    t0 = time.perf_counter()
    try:
        resp = gm.generate_content(full_prompt, generation_config=gen_config)
        elapsed = time.perf_counter() - t0
        text = resp.text or ""

        # Token usage from old SDK
        usage = {}
        if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
            um = resp.usage_metadata
            usage = {
                'prompt_tokens': getattr(um, 'prompt_token_count', None),
                'output_tokens': getattr(um, 'candidates_token_count', None),
                'thinking_tokens': getattr(um, 'thoughts_token_count', None),
                'total_tokens': getattr(um, 'total_token_count', None),
            }

        return {
            'sdk': 'google-generativeai (old)',
            'label': label,
            'ok': True,
            'elapsed': elapsed,
            'output_len': len(text),
            'usage': usage,
            'snippet': text[:200],
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'sdk': 'google-generativeai (old)',
            'label': label,
            'ok': False,
            'elapsed': elapsed,
            'error': f"{type(e).__name__}: {e}",
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_benchmark(model, runs=1, skip_new=False, skip_old=False):
    """Run the full benchmark suite."""
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('LLM_API_KEY')
    if not api_key:
        print("ERROR: No GEMINI_API_KEY or LLM_API_KEY in environment.")
        print("  Set it in your .env file or export it.")
        sys.exit(1)

    test_configs = select_configs(model)

    print(f"\n{'=' * 100}")
    print(f"  GEMINI TIMING BENCHMARK")
    print(f"{'=' * 100}")
    print(f"  Model:        {model}")
    print(f"  Runs/config:  {runs}")
    print(f"  New SDK:      {'YES' if NEW_SDK_AVAILABLE and not skip_new else 'SKIP'}")
    print(f"  Old SDK:      {'YES' if OLD_SDK_AVAILABLE and not skip_old else 'SKIP'}")
    print(f"  Test configs: {len(test_configs)}")
    print(f"  Sample path:  {SAMPLE_PATH}")
    print(f"{'=' * 100}\n")

    # ── Warmup ──
    print("  Warmup call (priming connection)...", end=" ", flush=True)
    if NEW_SDK_AVAILABLE and not skip_new:
        run_new_sdk(model, api_key, "warmup", None, False)
    elif OLD_SDK_AVAILABLE and not skip_old:
        run_old_sdk(model, api_key, "warmup", None, False)
    print("done.\n")

    # ── Run tests sequentially ──
    results = []
    total_tests = 0

    sdks_to_run = []
    if NEW_SDK_AVAILABLE and not skip_new:
        sdks_to_run.append(('new', run_new_sdk))
    if OLD_SDK_AVAILABLE and not skip_old:
        sdks_to_run.append(('old', run_old_sdk))

    for sdk_tag, runner_fn in sdks_to_run:
        sdk_label = "google-genai (new)" if sdk_tag == 'new' else "google-generativeai (old)"
        print(f"  --- {sdk_label} ---\n")

        # Old SDK does NOT support ThinkingConfig; only run non-thinking configs
        if sdk_tag == 'old':
            sdk_configs = [(l, tk, uj) for l, tk, uj in test_configs if tk is None]
            if not sdk_configs:
                sdk_configs = [("No thinking config (default)", None, False),
                               ("JSON + No thinking config", None, True)]
            print(f"    (Old SDK: ThinkingConfig not supported, running {len(sdk_configs)} non-thinking configs only)\n")
        else:
            sdk_configs = test_configs

        for label, thinking_kwargs, use_json in sdk_configs:
            timings = []
            last_result = None

            for r in range(runs):
                total_tests += 1
                result = runner_fn(model, api_key, label, thinking_kwargs, use_json)
                last_result = result
                timings.append(result['elapsed'])

                status = "OK" if result.get('ok') else "FAIL"
                thinking_tok = result.get('usage', {}).get('thinking_tokens', '')
                thinking_str = f" | think_tok={thinking_tok}" if thinking_tok else ""
                print(f"    [{status}] {label:40s} | {result['elapsed']:6.2f}s{thinking_str}")

                if not result.get('ok'):
                    err = result.get('error', '?')
                    print(f"           Error: {err[:80]}")

                # Small delay between API calls to avoid rate-limiting
                time.sleep(0.5)

            # Store aggregate result
            avg_time = sum(timings) / len(timings) if timings else 0
            min_time = min(timings) if timings else 0
            results.append({
                'sdk': last_result['sdk'] if last_result else sdk_label,
                'label': label,
                'ok': last_result.get('ok', False) if last_result else False,
                'avg_time': avg_time,
                'min_time': min_time,
                'runs': len(timings),
                'usage': last_result.get('usage', {}) if last_result else {},
                'error': last_result.get('error') if last_result and not last_result.get('ok') else None,
            })

        print()

    # ── Summary Table ──
    print(f"\n{'=' * 100}")
    print(f"{'RESULTS SUMMARY':^100}")
    print(f"{'=' * 100}")
    hdr = (f"  {'SDK':<26} | {'Configuration':<40} | {'Avg':>6} | {'Min':>6} | "
           f"{'Think':>6} | {'Status'}")
    print(hdr)
    print(f"  {'-' * 96}")

    # Sort: successful first (by avg_time), then failed
    ok_results = [r for r in results if r['ok']]
    fail_results = [r for r in results if not r['ok']]

    for r in sorted(ok_results, key=lambda x: x['avg_time']):
        think = r.get('usage', {}).get('thinking_tokens', '')
        think_str = str(think) if think else '-'
        print(f"  {r['sdk']:<26} | {r['label']:<40} | {r['avg_time']:5.2f}s | "
              f"{r['min_time']:5.2f}s | {think_str:>6} | OK")

    for r in fail_results:
        err_short = (r.get('error') or '?')[:40]
        print(f"  {r['sdk']:<26} | {r['label']:<40} | {'N/A':>6} | "
              f"{'N/A':>6} | {'N/A':>6} | FAIL: {err_short}")

    print(f"{'=' * 100}")

    # ── Winner ──
    if ok_results:
        fastest = min(ok_results, key=lambda x: x['avg_time'])
        print(f"\n  FASTEST: {fastest['sdk']} | {fastest['label']} | "
              f"avg {fastest['avg_time']:.2f}s | min {fastest['min_time']:.2f}s")

        # Compare new vs old if both available
        new_ok = [r for r in ok_results if 'new' in r['sdk']]
        old_ok = [r for r in ok_results if 'old' in r['sdk']]
        if new_ok and old_ok:
            best_new = min(new_ok, key=lambda x: x['avg_time'])
            best_old = min(old_ok, key=lambda x: x['avg_time'])
            diff = best_new['avg_time'] - best_old['avg_time']
            if diff > 0:
                print(f"\n  Old SDK is faster by {diff:.2f}s "
                      f"(best old: {best_old['avg_time']:.2f}s vs best new: {best_new['avg_time']:.2f}s)")
                print(f"    Old: {best_old['label']}")
                print(f"    New: {best_new['label']}")
            else:
                print(f"\n  New SDK is faster by {-diff:.2f}s "
                      f"(best new: {best_new['avg_time']:.2f}s vs best old: {best_old['avg_time']:.2f}s)")
                print(f"    New: {best_new['label']}")
                print(f"    Old: {best_old['label']}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Gemini thinking configurations across SDKs"
    )
    parser.add_argument(
        '--model', '-m',
        default=os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash'),
        help='Gemini model name (default: GEMINI_MODEL env var or gemini-2.0-flash)',
    )
    parser.add_argument(
        '--runs', '-r', type=int, default=1,
        help='Number of runs per configuration (default: 1, use 3+ for reliable averages)',
    )
    parser.add_argument(
        '--only-new', action='store_true',
        help='Only test the new google-genai SDK',
    )
    parser.add_argument(
        '--only-old', action='store_true',
        help='Only test the old google-generativeai SDK',
    )
    args = parser.parse_args()

    run_benchmark(
        model=args.model,
        runs=args.runs,
        skip_new=args.only_old,
        skip_old=args.only_new,
    )


if __name__ == '__main__':
    main()
