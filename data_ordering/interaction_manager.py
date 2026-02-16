"""
Interaction Manager for the data ordering tool.
Handles manual decision points during pipeline processing.

Supports three modes:
- Interactive: Pauses and prompts user for decisions on ambiguous cases
- Deferred: Automatically moves ambiguous cases to review folders
- Step-by-step: Pauses at EVERY major pipeline stage for manual verification
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions that can be requested."""
    NUMERIC_ID_AMBIGUITY = "numeric_id_ambiguity"  # Is 12345 a specimen ID or camera counter?
    METADATA_CONFLICT = "metadata_conflict"         # Path says X, filename says Y
    SOURCE_DISCREPANCY = "source_discrepancy"       # Different extraction sources disagree on a field
    CAMERA_NUMBER_FLAG = "camera_number_flag"       # A numeric ID was flagged as likely camera-generated
    DUPLICATE_DETECTED = "duplicate_detected"       # File is duplicate of another
    DUPLICATE_METADATA_DISCREPANCY = "duplicate_metadata_discrepancy"  # Duplicates have different metadata
    FILENAME_COLLISION = "filename_collision"       # Two files with same destination name
    UNKNOWN_COLLECTION = "unknown_collection"       # Cannot determine collection
    NO_SPECIMEN_ID = "no_specimen_id"               # Could not extract specimen ID
    LLM_REGEX_FAILED = "llm_regex_failed"           # LLM-provided regex didn't work


class DecisionOutcome(Enum):
    """Possible outcomes from a decision."""
    ACCEPT = "accept"           # Use the suggested/first option
    REJECT = "reject"           # Reject and use alternative
    SKIP = "skip"               # Skip this file
    DEFER = "defer"             # Move to review folder for later
    CUSTOM = "custom"           # User provided custom value
    MERGE = "merge"             # Merge metadata from multiple sources
    APPLY_TO_SUBDIRECTORY = "apply_to_subdirectory"  # Apply same choice to all files in subdirectory


@dataclass
class DecisionRequest:
    """A request for user decision."""
    decision_type: DecisionType
    file_path: Path
    context: Dict[str, Any]  # Type-specific context
    options: List[str] = field(default_factory=list)
    default_option: int = 0  # Index of default option
    message: str = ""
    
    def to_dict(self) -> dict:
        return {
            'decision_type': self.decision_type.value,
            'file_path': str(self.file_path),
            'context': self.context,
            'options': self.options,
            'default_option': self.default_option,
            'message': self.message,
            'timestamp': datetime.now().isoformat(),
        }


@dataclass
class DecisionResult:
    """Result of a user decision."""
    outcome: DecisionOutcome
    selected_option: Optional[int] = None
    custom_value: Optional[str] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'outcome': self.outcome.value,
            'selected_option': self.selected_option,
            'custom_value': self.custom_value,
            'notes': self.notes,
        }


class InteractionMode(Enum):
    """Mode of operation for the interaction manager."""
    INTERACTIVE = "interactive"      # Pause only on ambiguous decisions
    DEFERRED = "deferred"            # Auto-defer to review folders
    AUTO_ACCEPT = "auto_accept"      # Auto-accept defaults (fastest, least safe)
    STEP_BY_STEP = "step_by_step"    # Pause at EVERY major stage for verification


class InteractionManager:
    """
    Manages user interactions and decisions during pipeline processing.
    """
    
    def __init__(
        self,
        mode: InteractionMode = InteractionMode.DEFERRED,
        review_base_dir: Optional[Path] = None,
        log_decisions: bool = True,
    ):
        """
        Initialize the interaction manager.
        
        Args:
            mode: How to handle decision points
            review_base_dir: Directory for deferred review items
            log_decisions: Whether to log all decisions to a file
        """
        self.mode = mode
        self.review_base_dir = review_base_dir
        self.log_decisions = log_decisions
        
        # Track decisions made
        self.decisions_made: List[Tuple[DecisionRequest, DecisionResult]] = []
        self.deferred_items: List[DecisionRequest] = []
        
        # Statistics
        self.stats = {
            'interactive_decisions': 0,
            'auto_deferred': 0,
            'auto_accepted': 0,
            'stage_confirmations': 0,
        }
    
    def request_decision(self, request: DecisionRequest) -> DecisionResult:
        """
        Request a decision from the user or apply automatic logic.
        
        Args:
            request: The decision request
            
        Returns:
            DecisionResult with the outcome
        """
        if self.mode == InteractionMode.INTERACTIVE or self.mode == InteractionMode.STEP_BY_STEP:
            result = self._interactive_decision(request)
            self.stats['interactive_decisions'] += 1
        elif self.mode == InteractionMode.DEFERRED:
            result = self._deferred_decision(request)
            self.stats['auto_deferred'] += 1
        else:  # AUTO_ACCEPT
            result = self._auto_accept_decision(request)
            self.stats['auto_accepted'] += 1
        
        # Log decision
        self.decisions_made.append((request, result))
        
        if self.log_decisions:
            logger.info(f"Decision: {request.decision_type.value} -> {result.outcome.value}")
        
        return result
    
    def _interactive_decision(self, request: DecisionRequest) -> DecisionResult:
        """Handle decision interactively via CLI."""
        print("\n" + "=" * 60)
        print(f"DECISION REQUIRED: {request.decision_type.value}")
        print("=" * 60)
        print(f"File: {request.file_path}")
        print(f"\n{request.message}")
        
        is_dup_meta = (request.decision_type == DecisionType.DUPLICATE_METADATA_DISCREPANCY)
        supports_subdir = (request.decision_type in (
            DecisionType.SOURCE_DISCREPANCY,
            DecisionType.CAMERA_NUMBER_FLAG,
        ))
        
        # Show context only for non-table decision types (table is already in the message)
        if request.context and not is_dup_meta:
            print("\nContext:")
            for key, value in request.context.items():
                # Skip internal keys used by the reconciliation engine
                if key.startswith('_'):
                    continue
                print(f"  {key}: {value}")
        
        print("\nOptions:")
        for i, option in enumerate(request.options):
            print(f"  [{i}] {option}")
        
        print(f"  [d] Defer to manual review folder")
        if not is_dup_meta:
            print(f"  [s] Skip this file")
            print(f"  [c] Enter custom value")
        if supports_subdir:
            print(f"  [a] Apply chosen option to ALL files in this subdirectory (same field & sources)")
        
        if is_dup_meta:
            valid_keys = f"0-{len(request.options)-1}, d"
        elif supports_subdir:
            valid_keys = f"0-{len(request.options)-1}, d, s, c, a"
        else:
            valid_keys = f"0-{len(request.options)-1}, d, s, c"
        
        while True:
            try:
                choice = input(f"\nEnter choice ({valid_keys}): ").strip().lower()
                
                if choice == '':
                    print("Please select an option explicitly.")
                    continue
                elif choice == 'd':
                    self.deferred_items.append(request)
                    return DecisionResult(outcome=DecisionOutcome.DEFER)
                elif choice == 's' and not is_dup_meta:
                    return DecisionResult(outcome=DecisionOutcome.SKIP)
                elif choice == 'c' and not is_dup_meta:
                    custom = input("Enter custom value: ").strip()
                    subdir_too = False
                    if supports_subdir:
                        subdir_choice = input("Apply this custom value to the whole subdirectory? [y/N]: ").strip().lower()
                        subdir_too = subdir_choice in ('y', 'yes')
                    return DecisionResult(
                        outcome=DecisionOutcome.APPLY_TO_SUBDIRECTORY if subdir_too else DecisionOutcome.CUSTOM,
                        custom_value=custom,
                    )
                elif choice == 'a' and supports_subdir:
                    # Ask which numbered option to apply to the whole subdirectory
                    sub_choice = input(f"  Which option to apply to the whole subdirectory? [0-{len(request.options)-1}]: ").strip()
                    if sub_choice.isdigit():
                        idx = int(sub_choice)
                        if 0 <= idx < len(request.options):
                            return DecisionResult(
                                outcome=DecisionOutcome.APPLY_TO_SUBDIRECTORY,
                                selected_option=idx,
                            )
                    print("Invalid sub-option. Try again.")
                    continue
                elif choice.isdigit():
                    idx = int(choice)
                    if 0 <= idx < len(request.options):
                        # For dup-metadata, the last numbered option is
                        # "Enter custom value per field" — handle inline.
                        if is_dup_meta and idx == len(request.options) - 1:
                            return self._prompt_custom_per_field(request)
                        return DecisionResult(
                            outcome=DecisionOutcome.ACCEPT,
                            selected_option=idx,
                        )
                    else:
                        print(f"Invalid option. Choose {valid_keys}")
                else:
                    print("Invalid input. Try again.")
            except KeyboardInterrupt:
                print("\nInterrupted. Deferring this item.")
                self.deferred_items.append(request)
                return DecisionResult(outcome=DecisionOutcome.DEFER)
    
    def _prompt_custom_per_field(self, request: DecisionRequest) -> DecisionResult:
        """Sub-prompt for entering a custom value per discrepant field.
        
        Shows each discrepant field with its current values across files,
        then asks the user to type the desired value.
        """
        discrepancies = request.context.get('discrepancies', {})
        file_paths = request.context.get('file_paths', [])
        file_names = [Path(fp).name for fp in file_paths]
        
        print("\n--- Enter custom value for each field ---")
        custom_values = {}
        for field_name, entries in discrepancies.items():
            values_display = ", ".join(
                f"[{i}] {file_names[i]}={val}" for i, (_, val) in enumerate(entries)
            )
            print(f"\n  {field_name}: {values_display}")
            val = input(f"  Enter value for '{field_name}': ").strip()
            if val:
                custom_values[field_name] = val
        
        # Encode as "field=value; field2=value2" for the orchestrator
        custom_str = "; ".join(f"{k}={v}" for k, v in custom_values.items())
        return DecisionResult(
            outcome=DecisionOutcome.CUSTOM,
            custom_value=custom_str,
        )
    
    def _deferred_decision(self, request: DecisionRequest) -> DecisionResult:
        """Automatically defer to review folder."""
        self.deferred_items.append(request)
        return DecisionResult(outcome=DecisionOutcome.DEFER)
    
    def _auto_accept_decision(self, request: DecisionRequest) -> DecisionResult:
        """Automatically accept default option."""
        return DecisionResult(
            outcome=DecisionOutcome.ACCEPT,
            selected_option=request.default_option,
        )
    
    def confirm_stage(
        self,
        stage_name: str,
        stage_description: str,
        summary_data: Dict[str, Any] = None,
        sample_items: List[Dict[str, Any]] = None,
        max_samples: int = 10,
    ) -> bool:
        """
        Request user confirmation before proceeding to next stage.
        Only prompts in STEP_BY_STEP mode.
        
        Args:
            stage_name: Name of the completed stage (e.g., "SCANNING")
            stage_description: Description of what the stage did
            summary_data: Key metrics/counts from the stage
            sample_items: Sample of processed items for review
            max_samples: Maximum number of samples to display
            
        Returns:
            True to continue, False to abort
        """
        if self.mode != InteractionMode.STEP_BY_STEP:
            return True  # Auto-continue in other modes
        
        print("\n")
        print("╔" + "═" * 68 + "╗")
        print(f"║ STAGE COMPLETE: {stage_name:<50} ║")
        print("╠" + "═" * 68 + "╣")
        print(f"║ {stage_description:<66} ║")
        print("╚" + "═" * 68 + "╝")
        
        # Show summary data
        if summary_data:
            print("\n📊 Summary:")
            for key, value in summary_data.items():
                print(f"   • {key}: {value}")
        
        # Show sample items
        if sample_items:
            print(f"\n📋 Sample items (showing {min(len(sample_items), max_samples)} of {len(sample_items)}):")
            for i, item in enumerate(sample_items[:max_samples]):
                print(f"\n   [{i+1}]")
                for key, value in item.items():
                    if value is not None:
                        val_str = str(value)
                        if len(val_str) > 60:
                            val_str = val_str[:57] + "..."
                        print(f"       {key}: {val_str}")
        
        print("\n" + "-" * 70)
        print("Options:")
        print("   [Enter/y] Continue to next stage")
        print("   [n/q]     Abort pipeline (state will be saved)")
        print("   [s]       Save current state and continue")
        print("   [r]       Review more samples")
        print("-" * 70)
        
        while True:
            try:
                choice = input("\nContinue? [y]: ").strip().lower()
                
                if choice in ('', 'y', 'yes'):
                    self.stats['stage_confirmations'] += 1
                    return True
                elif choice in ('n', 'no', 'q', 'quit', 'abort'):
                    print("\n⚠️  Aborting pipeline. State will be saved for resume.")
                    return False
                elif choice == 's':
                    print("   State will be saved automatically.")
                    self.stats['stage_confirmations'] += 1
                    return True
                elif choice == 'r':
                    if sample_items and len(sample_items) > max_samples:
                        print(f"\n📋 All items ({len(sample_items)} total):")
                        for i, item in enumerate(sample_items):
                            print(f"\n   [{i+1}]")
                            for key, value in item.items():
                                if value is not None:
                                    print(f"       {key}: {value}")
                    else:
                        print("   No additional items to show.")
                else:
                    print("   Invalid choice. Enter y/n/s/r")
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Aborting pipeline.")
                return False
    
    def is_step_by_step(self) -> bool:
        """Check if running in step-by-step mode."""
        return self.mode == InteractionMode.STEP_BY_STEP
    
    def get_review_folder(self, decision_type: DecisionType) -> Path:
        """
        Get the review folder for a specific decision type.
        
        Args:
            decision_type: Type of decision
            
        Returns:
            Path to the review subfolder
        """
        if not self.review_base_dir:
            raise ValueError("review_base_dir not set")
        
        folder_map = {
            DecisionType.NUMERIC_ID_AMBIGUITY: "Ambiguous_IDs",
            DecisionType.METADATA_CONFLICT: "Metadata_Conflicts",
            DecisionType.SOURCE_DISCREPANCY: "Source_Discrepancies",
            DecisionType.CAMERA_NUMBER_FLAG: "Camera_Number_Review",
            DecisionType.DUPLICATE_DETECTED: "Duplicates_Review",
            DecisionType.DUPLICATE_METADATA_DISCREPANCY: "Duplicate_Metadata_Review",
            DecisionType.FILENAME_COLLISION: "Filename_Collisions",
            DecisionType.UNKNOWN_COLLECTION: "Unknown_Collection",
            DecisionType.NO_SPECIMEN_ID: "No_Specimen_ID",
            DecisionType.LLM_REGEX_FAILED: "LLM_Review",
        }
        
        subfolder = folder_map.get(decision_type, "Other_Review")
        path = self.review_base_dir / subfolder
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def save_deferred_log(self, output_path: Path):
        """
        Save a log of all deferred items for later review.
        
        Args:
            output_path: Path to save the JSON log
        """
        import json
        
        data = {
            'generated_at': datetime.now().isoformat(),
            'mode': self.mode.value,
            'stats': self.stats,
            'deferred_items': [req.to_dict() for req in self.deferred_items],
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.deferred_items)} deferred items to {output_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all decisions made."""
        return {
            'mode': self.mode.value,
            'total_decisions': len(self.decisions_made),
            'deferred_count': len(self.deferred_items),
            'stats': self.stats,
            'by_type': self._count_by_type(),
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count decisions by type."""
        counts = {}
        for req, _ in self.decisions_made:
            key = req.decision_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts


# === Convenience functions for common decision types ===

def create_numeric_id_decision(
    file_path: Path,
    numeric_value: str,
    sample_filenames: List[str],
) -> DecisionRequest:
    """Create a decision request for ambiguous numeric IDs."""
    return DecisionRequest(
        decision_type=DecisionType.NUMERIC_ID_AMBIGUITY,
        file_path=file_path,
        message=f"Found numeric ID '{numeric_value}' in filename. Is this a specimen ID or a camera-generated number?",
        context={
            'numeric_value': numeric_value,
            'sample_filenames': sample_filenames[:5],
        },
        options=[
            f"'{numeric_value}' is a SPECIMEN ID - use it",
            f"'{numeric_value}' is a CAMERA NUMBER - ignore it",
            "Cannot determine - need more context",
        ],
        default_option=1,  # Default to camera number (safer)
    )


def create_metadata_conflict_decision(
    file_path: Path,
    field_name: str,
    value_from_path: Any,
    value_from_filename: Any,
) -> DecisionRequest:
    """Create a decision request for conflicting metadata."""
    return DecisionRequest(
        decision_type=DecisionType.METADATA_CONFLICT,
        file_path=file_path,
        message=f"Conflict in '{field_name}': path says '{value_from_path}', filename says '{value_from_filename}'",
        context={
            'field_name': field_name,
            'value_from_path': str(value_from_path),
            'value_from_filename': str(value_from_filename),
        },
        options=[
            f"Use value from PATH: {value_from_path}",
            f"Use value from FILENAME: {value_from_filename}",
            "Use BOTH (if applicable)",
        ],
        default_option=0,  # Default to path (usually more reliable)
    )


def create_duplicate_decision(
    file_path: Path,
    original_path: Path,
    similarity: float,
) -> DecisionRequest:
    """Create a decision request for detected duplicates."""
    return DecisionRequest(
        decision_type=DecisionType.DUPLICATE_DETECTED,
        file_path=file_path,
        message=f"File appears to be a duplicate (similarity: {similarity:.1%})",
        context={
            'original_path': str(original_path),
            'similarity': similarity,
        },
        options=[
            "SKIP this file (keep original only)",
            "KEEP BOTH files",
            "REPLACE original with this file",
            "MERGE metadata and keep one",
        ],
        default_option=0,  # Default to skip duplicate
    )


def create_duplicate_metadata_decision(
    file_paths: List[Path],
    discrepancies: Dict[str, List],
) -> DecisionRequest:
    """Create a decision request for duplicates with differing metadata.
    
    Args:
        file_paths: All files in the duplicate group
        discrepancies: Dict mapping field_name -> list of (file_path, value) pairs
    """
    # --- Build a comparison table ---
    file_names = [Path(fp).name for fp in file_paths]
    field_names = list(discrepancies.keys())
    
    # Build a value lookup:  field -> {file_path_str -> value}
    value_map: Dict[str, Dict[str, str]] = {}
    for field_name, entries in discrepancies.items():
        value_map[field_name] = {}
        for fpath_str, val in entries:
            value_map[field_name][str(fpath_str)] = val
    
    # Determine column widths for the table
    field_header = "Field"
    col_width_field = max(len(field_header), *(len(f) for f in field_names))
    
    col_widths = []
    for i, fp in enumerate(file_paths):
        fname = file_names[i]
        header_label = f"[{i}] {fname}"
        max_val_len = 0
        for field_name in field_names:
            val = value_map[field_name].get(str(fp), "(empty)")
            max_val_len = max(max_val_len, len(val))
        col_widths.append(max(len(header_label), max_val_len))
    
    # Build the table
    sep = "─"
    lines = []
    
    # Header
    header_parts = [f" {field_header:<{col_width_field}} "]
    for i, fp in enumerate(file_paths):
        header_label = f"[{i}] {file_names[i]}"
        header_parts.append(f" {header_label:<{col_widths[i]}} ")
    
    divider_parts = [sep * (col_width_field + 2)]
    for w in col_widths:
        divider_parts.append(sep * (w + 2))
    
    lines.append("┌" + "┬".join(divider_parts) + "┐")
    lines.append("│" + "│".join(header_parts) + "│")
    lines.append("├" + "┼".join(divider_parts) + "┤")
    
    # Data rows — one per discrepant field
    for field_name in field_names:
        row_parts = [f" {field_name:<{col_width_field}} "]
        for i, fp in enumerate(file_paths):
            val = value_map[field_name].get(str(fp), "(empty)")
            row_parts.append(f" {val:<{col_widths[i]}} ")
        lines.append("│" + "│".join(row_parts) + "│")
    
    lines.append("└" + "┴".join(divider_parts) + "┘")
    table_text = "\n".join(lines)
    
    # Build options: one per file (use that file's metadata) + custom
    options = []
    for i, fpath in enumerate(file_paths):
        options.append(f"Use metadata from [{i}]: {Path(fpath).name}")
    options.append("Enter custom value for each discrepant field")
    
    return DecisionRequest(
        decision_type=DecisionType.DUPLICATE_METADATA_DISCREPANCY,
        file_path=file_paths[0],  # Primary file
        message=(
            f"Duplicate group ({len(file_paths)} files) has metadata discrepancies\n"
            f"Discrepant fields: {', '.join(field_names)}\n\n"
            f"{table_text}"
        ),
        context={
            'file_paths': [str(p) for p in file_paths],
            'discrepancies': {
                k: [(str(p), v) for p, v in vals]
                for k, vals in discrepancies.items()
            },
        },
        options=options,
        default_option=0,  # Default to first file (the "original")
    )


def create_unknown_collection_decision(
    file_path: Path,
    detected_info: Dict[str, Any],
) -> DecisionRequest:
    """Create a decision request for files with unknown collection."""
    return DecisionRequest(
        decision_type=DecisionType.UNKNOWN_COLLECTION,
        file_path=file_path,
        message="Could not determine which collection this file belongs to.",
        context=detected_info,
        options=[
            "Las Hoyas (LH)",
            "Buenache (BUE)",
            "Montsec (MON)",
            "Unknown - move to review",
        ],
        default_option=3,  # Default to unknown
    )


# ---------- Source-discrepancy & camera-number decisions ----------

# Human-friendly labels for the three extraction sources
SOURCE_LABELS = {
    'path_llm': 'Path LLM',
    'filename_llm': 'Filename LLM',
    'pattern_extractor': 'Pattern Extractor',
}


def create_source_discrepancy_decision(
    file_path: Path,
    field_name: str,
    values_by_source: Dict[str, Any],
    directory: str,
) -> DecisionRequest:
    """Create a decision request when extraction sources disagree on a field.

    Args:
        file_path: The file whose metadata is in conflict.
        field_name: Metadata field name (e.g. 'specimen_id', 'campaign_year').
        values_by_source: Mapping of source key → extracted value.
        directory: The subdirectory path (used for subdirectory-wide caching).

    Returns:
        DecisionRequest with one numbered option per source value, plus a
        custom-value option.  The user can also choose [a] to apply the
        decision to the entire subdirectory.
    """
    # Build a readable table
    lines = []
    source_keys = list(values_by_source.keys())
    for src_key in source_keys:
        label = SOURCE_LABELS.get(src_key, src_key)
        lines.append(f"  {label}: {values_by_source[src_key]}")
    value_table = "\n".join(lines)

    options = []
    for src_key in source_keys:
        label = SOURCE_LABELS.get(src_key, src_key)
        options.append(f"Use {label} value: {values_by_source[src_key]}")
    options.append("Leave field empty")
    options.append("Enter a custom value")

    return DecisionRequest(
        decision_type=DecisionType.SOURCE_DISCREPANCY,
        file_path=file_path,
        message=(
            f"Discrepancy on field '{field_name}' for file {file_path.name}:\n"
            f"{value_table}\n\n"
            f"Choose which value to keep, or press [a] to apply the same "
            f"choice to every file in this subdirectory with the same conflict."
        ),
        context={
            'field_name': field_name,
            'values_by_source': values_by_source,
            'directory': directory,
            # internal keys prefixed with _ are hidden in the CLI context display
            '_source_keys': source_keys,
        },
        options=options,
        default_option=0,
    )


def create_camera_number_decision(
    file_path: Path,
    numeric_id: str,
    raw_match: str,
    directory: str,
) -> DecisionRequest:
    """Create a decision request for a numeric ID flagged as likely camera-generated.

    Args:
        file_path: The file containing the flagged number.
        numeric_id: The numeric value extracted.
        raw_match: The raw text that matched the camera pattern.
        directory: The subdirectory path (for subdirectory-wide caching).

    Returns:
        DecisionRequest.  The user can choose to use the value as-is, discard
        it, enter a custom value, or apply the decision to the whole
        subdirectory.
    """
    return DecisionRequest(
        decision_type=DecisionType.CAMERA_NUMBER_FLAG,
        file_path=file_path,
        message=(
            f"The numeric ID '{numeric_id}' (matched as '{raw_match}') in file "
            f"'{file_path.name}' looks like a camera-generated number.\n\n"
            f"Choose what to do, or press [a] to apply the same choice to all "
            f"files in this subdirectory with the same flag."
        ),
        context={
            'numeric_id': numeric_id,
            'raw_match': raw_match,
            'directory': directory,
        },
        options=[
            f"Use '{numeric_id}' as specimen ID anyway",
            "Discard — leave specimen ID empty",
            "Enter a custom specimen ID",
        ],
        default_option=1,  # Default: discard (safer)
    )
