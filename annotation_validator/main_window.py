"""
Main window for the Annotation Validator application.

A visual, scrollable single-page layout:
  ┌──────────────────────────────────────────────────────────────┐
  │  Header bar:   Sample X / N       [Save]  [Next →]          │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  📂 Original Paths                                           │
  │   1. D:/MUPA/Insecta/Las_Hoyas/img.orf                      │
  │                                                              │
  │  ┌──────────────────┐  ┌──────────────────────────────────┐  │
  │  │                  │  │ 📋 Annotation Fields              │  │
  │  │   MAIN IMAGE     │  │                                  │  │
  │  │   (large)        │  │  Specimen ID    LH-12345         │  │
  │  │                  │  │  Macroclass     Insects           │  │
  │  │                  │  │  Class          Insecta           │  │
  │  └──────────────────┘  │  Year           2009              │  │
  │                        │  Source          MUPA              │  │
  │                        └──────────────────────────────────┘  │
  │                                                              │
  │  🔄 Duplicates (2)                                           │
  │  ┌──────┐ ┌──────────────────────────────────────────────┐   │
  │  │ img  │ │ specimen_id  LH-12345 = LH-12345  ✓         │   │
  │  │      │ │ year         2009     ≠ 2011       ✗         │   │
  │  └──────┘ └──────────────────────────────────────────────┘   │
  │                                                              │
  ├──────────────────────────────────────────────────────────────┤
  │  Notes: [__________]  [✓ OK]  [✗ Bad]  [? Skip]  [Next →]  │
  └──────────────────────────────────────────────────────────────┘
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import pandas as pd

from data_loader import DataLoader, AnnotationSample, DuplicateInfo


RAW_EXTENSIONS = {'.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2', '.raw'}


# ═══════════════════════════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════════════════════════
C_BG       = '#1a1a2e'
C_BG2      = '#16213e'
C_CARD     = '#1f2940'
C_CARD_HL  = '#273552'
C_FG       = '#e0e0e0'
C_FG_DIM   = '#7a8ba5'
C_FG_BR    = '#ffffff'
C_ACCENT   = '#0078d4'
C_GREEN    = '#3fb950'
C_RED      = '#f85149'
C_AMBER    = '#d29922'
C_CYAN     = '#58a6ff'
C_ORANGE   = '#e8853d'
C_BORDER   = '#2a3a55'


# ═══════════════════════════════════════════════════════════════════
# Helper: Scrollable Frame
# ═══════════════════════════════════════════════════════════════════

class ScrollableFrame(tk.Frame):
    """A frame inside a canvas that scrolls vertically."""

    def __init__(self, parent, **kw):
        bg = kw.pop('bg', C_BG)
        super().__init__(parent, bg=bg, **kw)

        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)

        self.inner = tk.Frame(self.canvas, bg=bg)
        self.inner_id = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)

        self.inner.bind('<Configure>', self._on_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Bind mouse wheel (cross-platform)
        self.canvas.bind('<Enter>', self._bind_wheel)
        self.canvas.bind('<Leave>', self._unbind_wheel)

    def _on_configure(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.inner_id, width=event.width)

    def _bind_wheel(self, _):
        self.canvas.bind_all('<MouseWheel>',
                             lambda e: self.canvas.yview_scroll(-1 * (e.delta // 120), 'units'))

    def _unbind_wheel(self, _):
        self.canvas.unbind_all('<MouseWheel>')

    def scroll_to_top(self):
        self.canvas.yview_moveto(0)


# ═══════════════════════════════════════════════════════════════════
# MainWindow
# ═══════════════════════════════════════════════════════════════════

class MainWindow:
    """Visual annotation validator."""

    MAX_IMG_W = 800
    MAX_IMG_H = 650
    DUP_IMG_W = 300
    DUP_IMG_H = 240

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Annotation Validator")
        self.root.geometry("1300x900")
        self.root.minsize(950, 650)
        self.root.configure(bg=C_BG)

        self.data_loader: Optional[DataLoader] = None
        self.current_sample: Optional[AnnotationSample] = None
        self._photo_refs: List = []
        self._sample_number = 0

        self.notes_var = tk.StringVar(value="")
        self.stats_var = tk.StringVar(value="")
        self.sort_order_var = tk.StringVar(value="path+phash")
        self.dup_only_var = tk.BooleanVar(value=False)
        self.missing_only_var = tk.BooleanVar(value=False)
        self.multi_fossil_var = tk.BooleanVar(value=False)

        self._current_view: Optional[tk.Widget] = None
        self._setup_theme()
        self._show_welcome()

    # ─── Theme ────────────────────────────────────────────────────

    def _setup_theme(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('TFrame', background=C_BG)
        s.configure('TLabel', background=C_BG, foreground=C_FG)
        s.configure('TButton', background='#2a3a55', foreground=C_FG, padding=6,
                    font=('Segoe UI', 10))
        s.map('TButton',
              background=[('active', C_ACCENT)],
              foreground=[('active', C_FG_BR)])
        s.configure('Accent.TButton', background=C_ACCENT, foreground=C_FG_BR,
                    font=('Segoe UI', 11, 'bold'), padding=10)
        s.map('Accent.TButton',
              background=[('active', '#005fa3')])
        s.configure('Green.TButton', background='#1a4731', foreground=C_GREEN,
                    font=('Segoe UI', 10, 'bold'))
        s.map('Green.TButton', background=[('active', '#246b46')])
        s.configure('Red.TButton', background='#4a1a1a', foreground=C_RED,
                    font=('Segoe UI', 10, 'bold'))
        s.map('Red.TButton', background=[('active', '#6b2424')])
        s.configure('Amber.TButton', background='#4a3a1a', foreground=C_AMBER,
                    font=('Segoe UI', 10, 'bold'))
        s.map('Amber.TButton', background=[('active', '#6b5324')])

        s.configure('TCheckbutton', background=C_BG, foreground=C_FG)
        s.map('TCheckbutton', background=[('active', C_BG)])

    # ─── Selectable Label (read-only Entry that supports copy) ────

    def _selectable_label(self, parent, text, bg=C_CARD, fg=C_FG,
                          font=('Consolas', 10), width=None, cursor='arrow'):
        """Create a read-only Entry that looks like a label but allows
        text selection and Ctrl+C copying."""
        kw = dict(
            readonlybackground=bg, fg=fg, font=font,
            relief='flat', bd=0, highlightthickness=0,
            selectbackground=C_ACCENT, selectforeground=C_FG_BR,
            cursor=cursor,
        )
        if width is not None:
            kw['width'] = width
        entry = tk.Entry(parent, **kw)
        entry.insert(0, text)
        entry.configure(state='readonly')
        return entry

    # ─── View Management ──────────────────────────────────────────

    def _clear(self):
        if self._current_view:
            self._current_view.destroy()
            self._current_view = None
        self._photo_refs.clear()

    # ─── Welcome Screen ──────────────────────────────────────────

    def _show_welcome(self):
        self._clear()
        f = tk.Frame(self.root, bg=C_BG)
        f.pack(fill='both', expand=True)
        self._current_view = f

        center = tk.Frame(f, bg=C_BG)
        center.place(relx=0.5, rely=0.42, anchor='center')

        tk.Label(center, text="🔬  Annotation Validator", bg=C_BG, fg=C_FG_BR,
                 font=('Segoe UI', 22, 'bold')).pack(pady=(0, 8))
        tk.Label(center, text="Visually review annotations from the data ordering pipeline.",
                 bg=C_BG, fg=C_FG_DIM, font=('Segoe UI', 11)).pack(pady=(0, 30))

        ttk.Button(center, text="📁  Open Output Directory",
                   command=self._open_directory, style='Accent.TButton').pack(pady=5)

        # ── Sort order selector ──
        sort_frame = tk.Frame(center, bg=C_BG)
        sort_frame.pack(pady=(20, 0))
        tk.Label(sort_frame, text="Sort order:", bg=C_BG, fg=C_FG,
                 font=('Segoe UI', 10, 'bold')).pack(side='left', padx=(0, 10))
        for value, label in [('path+phash', '📂 Path + pHash'),
                              ('random',     '🎲 Random')]:
            tk.Radiobutton(
                sort_frame, text=label, variable=self.sort_order_var,
                value=value, bg=C_BG, fg=C_FG, activebackground=C_BG,
                activeforeground=C_FG_BR, selectcolor=C_BG2,
                font=('Segoe UI', 10), indicatoron=True,
            ).pack(side='left', padx=6)

        tk.Label(center, text=(
            "\nExpected structure:\n"
            "  output_dir/\n"
            "    registries/  (or registros/)\n"
            "      anotaciones.xlsx\n"
            "      hashes.xlsx\n"
            "    Duplicados/\n"
            "      duplicados_registro.xlsx\n"
            "    organized/ …"
        ), bg=C_BG, fg=C_FG_DIM, font=('Consolas', 9), justify='left').pack(pady=20)

    # ─── Open Directory ───────────────────────────────────────────

    def _open_directory(self):
        path = filedialog.askdirectory(title="Select merge output directory")
        if not path:
            return
        self.data_loader = DataLoader(Path(path), sort_order=self.sort_order_var.get())
        ok, msg = self.data_loader.load()
        if not ok:
            messagebox.showerror("Error", msg)
            self.data_loader = None
            return
        messagebox.showinfo("Loaded", msg)
        self._sample_number = 0
        self._build_main_ui()
        self._next_sample()

    def _toggle_sort_order(self):
        """Switch between sort orders and re-sort the sample list."""
        if not self.data_loader:
            return
        orders = DataLoader.SORT_ORDERS
        current = self.sort_order_var.get()
        idx = orders.index(current) if current in orders else 0
        new_order = orders[(idx + 1) % len(orders)]
        self.sort_order_var.set(new_order)

        self.data_loader.reorder(new_order)
        self._sample_number = 0

        # Update header label
        sort_icon = '📂' if new_order == 'path+phash' else '🎲'
        self._sort_label.configure(text=f"{sort_icon} {new_order}")

        self._next_sample()

    # ─── Main UI Skeleton ─────────────────────────────────────────

    def _build_main_ui(self):
        self._clear()
        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill='both', expand=True)
        self._current_view = outer

        # ── Header bar ──
        hdr = tk.Frame(outer, bg=C_BG2, height=44)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        self.stats_label = tk.Label(hdr, textvariable=self.stats_var, bg=C_BG2, fg=C_FG,
                                    font=('Segoe UI', 10), padx=16)
        self.stats_label.pack(side='left')

        ttk.Button(hdr, text="💾 Save", command=self._save_session).pack(side='right', padx=6, pady=6)
        ttk.Button(hdr, text="📁", command=self._open_directory).pack(side='right', padx=2, pady=6)

        ttk.Checkbutton(hdr, text="Missing fields", variable=self.missing_only_var).pack(side='right', padx=6)
        ttk.Checkbutton(hdr, text="Has duplicates", variable=self.dup_only_var).pack(side='right', padx=6)

        # Sort order indicator + toggle
        sort_icon = '📂' if self.sort_order_var.get() == 'path+phash' else '🎲'
        sort_text = f"{sort_icon} {self.sort_order_var.get()}"
        self._sort_label = tk.Label(hdr, text=sort_text, bg=C_BG2, fg=C_FG_DIM,
                                    font=('Segoe UI', 9), padx=4)
        self._sort_label.pack(side='right', padx=(0, 2))
        ttk.Button(hdr, text="🔀", width=3,
                   command=self._toggle_sort_order).pack(side='right', padx=2, pady=6)

        # ── Scrollable content area ──
        self.scroll_frame = ScrollableFrame(outer, bg=C_BG)
        self.scroll_frame.pack(fill='both', expand=True)

        # ── Bottom bar ──
        bot = tk.Frame(outer, bg=C_BG2, height=52)
        bot.pack(fill='x', side='bottom')
        bot.pack_propagate(False)
        self._build_bottom_bar(bot)

        self._bind_keys()

    def _build_bottom_bar(self, parent):
        """Build the verdict/notes bar at the bottom."""
        inner = tk.Frame(parent, bg=C_BG2)
        inner.pack(fill='x', padx=16, pady=8)

        tk.Label(inner, text="Notes:", bg=C_BG2, fg=C_FG_DIM,
                 font=('Segoe UI', 10)).pack(side='left')
        self.notes_entry = tk.Entry(inner, textvariable=self.notes_var,
                                    bg=C_CARD, fg=C_FG, insertbackground=C_FG,
                                    font=('Segoe UI', 10), relief='flat',
                                    highlightthickness=1, highlightcolor=C_ACCENT,
                                    highlightbackground=C_BORDER)
        self.notes_entry.pack(side='left', padx=(6, 20), fill='x', expand=True)

        self.multi_fossil_cb = ttk.Checkbutton(
            inner, text="🦴 Multi-fossil  (m)",
            variable=self.multi_fossil_var)
        self.multi_fossil_cb.pack(side='left', padx=(0, 14))

        ttk.Button(inner, text="✓ OK  (1)", command=lambda: self._mark('valid'),
                   style='Green.TButton').pack(side='left', padx=3)
        ttk.Button(inner, text="✗ Bad  (2)", command=lambda: self._mark('invalid'),
                   style='Red.TButton').pack(side='left', padx=3)
        ttk.Button(inner, text="? Unsure  (3)", command=lambda: self._mark('uncertain'),
                   style='Amber.TButton').pack(side='left', padx=3)

        ttk.Button(inner, text="Next →", command=self._next_sample,
                   style='Accent.TButton').pack(side='right', padx=(12, 0))
        ttk.Button(inner, text="← Prev", command=self._prev_sample,
                   style='TButton').pack(side='right', padx=(0, 0))

    # ─── Sample Navigation ────────────────────────────────────────

    def _next_sample(self):
        if not self.data_loader:
            return
        sample = self.data_loader.get_sample(
            dup_only=self.dup_only_var.get(),
            missing_only=self.missing_only_var.get(),
        )
        if sample is None:
            self._render_finished()
            return
        self._sample_number += 1
        self.current_sample = sample
        self.notes_var.set("")
        self.multi_fossil_var.set(False)
        self._render_sample(sample)
        self._update_stats()

    def _prev_sample(self):
        if not self.data_loader:
            return
        sample = self.data_loader.get_previous_sample()
        if sample is None:
            return  # already at the beginning
        self._sample_number = max(1, self._sample_number - 1)
        self.current_sample = sample
        self.notes_var.set("")
        self.multi_fossil_var.set(False)
        self._render_sample(sample)
        self._update_stats()

    def _mark(self, verdict: str):
        if not self.current_sample or not self.data_loader:
            return
        self.data_loader.mark_reviewed(
            self.current_sample.uuid, verdict,
            notes=self.notes_var.get().strip(),
            multi_fossil=self.multi_fossil_var.get(),
        )
        self._next_sample()

    def _update_stats(self):
        if not self.data_loader:
            return
        s = self.data_loader.get_stats()
        v = s['verdicts']
        self.stats_var.set(
            f"  Sample #{self._sample_number}  ·  "
            f"Reviewed {s['reviewed']}/{s['total']}  │  "
            f"✓ {v.get('valid', 0)}   ✗ {v.get('invalid', 0)}   ? {v.get('uncertain', 0)}  │  "
            f"With dups: {s['annotations_with_duplicates']}"
        )

    # ─── Render Sample ────────────────────────────────────────────

    def _render_sample(self, sample: AnnotationSample):
        """Populate the scrollable area with the sample's visual cards."""
        self._photo_refs.clear()
        parent = self.scroll_frame.inner

        # Destroy previous content
        for w in parent.winfo_children():
            w.destroy()

        # Add some top padding
        tk.Frame(parent, bg=C_BG, height=10).pack()

        # ═══ ORIGINAL PATHS (top) ═══
        self._render_paths_card(parent, sample)

        # ═══ IMAGE + FIELDS ═══
        top_row = tk.Frame(parent, bg=C_BG)
        top_row.pack(fill='x', padx=20, pady=(0, 6))

        self._render_main_image(top_row, sample)
        self._render_fields_card(top_row, sample)

        # ═══ DUPLICATES (bottom) ═══
        if sample.duplicates:
            self._render_duplicates_section(parent, sample)

        # Bottom padding
        tk.Frame(parent, bg=C_BG, height=20).pack()

        self.scroll_frame.scroll_to_top()

    # ─── Main Image Card ──────────────────────────────────────────

    def _render_main_image(self, parent, sample: AnnotationSample):
        """Render the main image as a card on the left."""
        card = tk.Frame(parent, bg=C_CARD, bd=0, highlightbackground=C_BORDER,
                        highlightthickness=1)
        card.pack(side='left', anchor='n', padx=(0, 10), pady=4)

        resolved = self.data_loader.resolve_path(sample.current_path)
        photo = self._load_photo(resolved, self.MAX_IMG_W, self.MAX_IMG_H)

        if photo:
            lbl = tk.Label(card, image=photo, bg=C_CARD, bd=0)
            lbl.pack(padx=6, pady=(6, 2))
            self._photo_refs.append(photo)
        else:
            lbl = tk.Label(card, text="⚠  Image not found\n\n" + (sample.current_path or "(no path)"),
                           bg='#2a1a1a', fg=C_RED, font=('Segoe UI', 11),
                           width=50, height=14, wraplength=400, justify='center')
            lbl.pack(padx=6, pady=6)

        # Image caption
        caption_text = sample.current_path.split('\\')[-1] if sample.current_path else "(no path)"
        if resolved and resolved.exists():
            size_kb = resolved.stat().st_size / 1024
            sz = f"{size_kb:.0f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
            caption_text += f"   ({sz})"
        self._selectable_label(card, caption_text, bg=C_CARD, fg=C_FG_DIM,
                              font=('Consolas', 8)).pack(fill='x', padx=6, pady=(0, 6))

    # ─── Fields Card ──────────────────────────────────────────────

    def _render_fields_card(self, parent, sample: AnnotationSample):
        """Render annotation fields as a card on the right of the image."""
        card = tk.Frame(parent, bg=C_CARD, bd=0, highlightbackground=C_BORDER,
                        highlightthickness=1)
        card.pack(side='left', fill='both', expand=True, anchor='n', pady=4)

        # Title
        tk.Label(card, text="📋  Annotation Fields", bg=C_CARD, fg=C_FG_BR,
                 font=('Segoe UI', 13, 'bold'), anchor='w').pack(fill='x', padx=14, pady=(12, 8))

        # Separator
        tk.Frame(card, bg=C_BORDER, height=1).pack(fill='x', padx=14)

        fields_to_show = [
            ('UUID',           'uuid'),
            ('Specimen ID',    'specimen_id'),
            ('Macroclass',     'macroclass_label'),
            ('Class',          'class_label'),
            ('Genera',         'genera_label'),
            ('Campaign Year',  'campaign_year'),
            ('Source (fuente)', 'fuente'),
            ('Comments',       'comentarios'),
        ]

        for label, key in fields_to_show:
            row = tk.Frame(card, bg=C_CARD)
            row.pack(fill='x', padx=14, pady=2)

            tk.Label(row, text=f"{label}:", bg=C_CARD, fg=C_CYAN,
                     font=('Segoe UI', 10, 'bold'), width=16, anchor='e').pack(side='left')

            val = sample.fields.get(key)
            if val is not None and str(val).strip() and str(val) != 'nan':
                val_text = str(val)
                val_fg = C_FG
            else:
                val_text = "(empty)"
                val_fg = '#555555'

            self._selectable_label(row, val_text, bg=C_CARD, fg=val_fg,
                                  font=('Consolas', 10)).pack(side='left', fill='x',
                                                              expand=True, padx=(8, 0))

        # Hash info (small, at bottom of card)
        if sample.hash_info:
            tk.Frame(card, bg=C_BORDER, height=1).pack(fill='x', padx=14, pady=(8, 4))
            tk.Label(card, text="🔗 Hashes", bg=C_CARD, fg=C_FG_DIM,
                     font=('Segoe UI', 9, 'bold'), anchor='w').pack(fill='x', padx=14)
            for hkey, hlabel in [('md5_hash', 'MD5'), ('phash', 'pHash')]:
                hval = sample.hash_info.get(hkey) or "(none)"
                hrow = tk.Frame(card, bg=C_CARD)
                hrow.pack(fill='x', padx=14)
                tk.Label(hrow, text=f"{hlabel}:", bg=C_CARD, fg=C_FG_DIM,
                         font=('Consolas', 8), width=6, anchor='e').pack(side='left')
                self._selectable_label(hrow, hval, bg=C_CARD, fg=C_FG_DIM,
                                      font=('Consolas', 8)).pack(side='left', fill='x',
                                                                  expand=True, padx=(4, 0))

        # Bottom padding
        tk.Frame(card, bg=C_CARD, height=10).pack()

    # ─── Original Paths Card ──────────────────────────────────────

    def _render_paths_card(self, parent, sample: AnnotationSample):
        if not sample.original_paths:
            return

        card = tk.Frame(parent, bg=C_CARD, bd=0, highlightbackground=C_BORDER,
                        highlightthickness=1)
        card.pack(fill='x', padx=20, pady=4)

        tk.Label(card, text="📂  Original Paths", bg=C_CARD, fg=C_FG_BR,
                 font=('Segoe UI', 11, 'bold'), anchor='w').pack(fill='x', padx=14, pady=(10, 4))
        tk.Frame(card, bg=C_BORDER, height=1).pack(fill='x', padx=14)

        for i, p in enumerate(sample.original_paths, 1):
            self._selectable_label(card, f"  {i}. {p}", bg=C_CARD, fg='#aaaaaa',
                                  font=('Consolas', 9)).pack(fill='x', padx=14, pady=1)

        tk.Frame(card, bg=C_CARD, height=8).pack()

    # ─── Duplicates Section ───────────────────────────────────────

    def _render_duplicates_section(self, parent, sample: AnnotationSample):
        """Render each duplicate as a card with image + field comparison."""
        header = tk.Frame(parent, bg=C_BG)
        header.pack(fill='x', padx=20, pady=(10, 2))
        tk.Label(header, text=f"🔄  Duplicates ({len(sample.duplicates)})",
                 bg=C_BG, fg=C_ORANGE, font=('Segoe UI', 13, 'bold'),
                 anchor='w').pack(side='left')

        compare_fields = [
            ('specimen_id',      'Specimen ID'),
            ('macroclass_label', 'Macroclass'),
            ('class_label',      'Class'),
            ('genera_label',     'Genera'),
            ('campaign_year',    'Year'),
            ('fuente',           'Source'),
            ('comentarios',      'Comments'),
        ]

        for i, dup in enumerate(sample.duplicates):
            card = tk.Frame(parent, bg=C_CARD, bd=0,
                            highlightbackground=C_ORANGE, highlightthickness=1)
            card.pack(fill='x', padx=20, pady=4)

            # Title bar
            title_bar = tk.Frame(card, bg='#2a2010')
            title_bar.pack(fill='x')
            self._selectable_label(title_bar,
                                  f"  Duplicate #{i + 1}   (uuid: {dup.uuid})",
                                  bg='#2a2010', fg=C_ORANGE,
                                  font=('Segoe UI', 10, 'bold')).pack(fill='x', padx=10, pady=4)

            content = tk.Frame(card, bg=C_CARD)
            content.pack(fill='x', padx=8, pady=6)

            # ── Left: duplicate image ──
            img_frame = tk.Frame(content, bg=C_CARD)
            img_frame.pack(side='left', anchor='n', padx=(4, 10))

            dup_resolved = self.data_loader.resolve_path(dup.current_path)
            dup_photo = self._load_photo(dup_resolved, self.DUP_IMG_W, self.DUP_IMG_H)

            if dup_photo:
                tk.Label(img_frame, image=dup_photo, bg=C_CARD).pack()
                self._photo_refs.append(dup_photo)
            else:
                tk.Label(img_frame, text="⚠ No image", bg='#2a1a1a', fg=C_RED,
                         font=('Segoe UI', 9), width=22, height=6).pack()

            # Small path caption
            dup_fname = dup.current_path.split('\\')[-1] if dup.current_path else "(no path)"
            self._selectable_label(img_frame, dup_fname, bg=C_CARD, fg=C_FG_DIM,
                                  font=('Consolas', 7),
                                  width=max(len(dup_fname), 30)).pack(pady=(2, 0))

            # ── Right: field comparison table ──
            table_frame = tk.Frame(content, bg=C_CARD)
            table_frame.pack(side='left', fill='both', expand=True, anchor='n')

            # Original path of duplicate
            self._selectable_label(table_frame,
                                  f"Original: {dup.original_path or '(none)'}",
                                  bg=C_CARD, fg=C_FG_DIM,
                                  font=('Consolas', 8)).pack(fill='x', pady=(0, 4))

            # Column headers
            hdr = tk.Frame(table_frame, bg=C_CARD)
            hdr.pack(fill='x')
            tk.Label(hdr, text="Field", bg=C_CARD, fg=C_FG_DIM,
                     font=('Segoe UI', 8, 'bold'), width=13, anchor='w').pack(side='left')
            tk.Label(hdr, text="Duplicate", bg=C_CARD, fg=C_FG_DIM,
                     font=('Segoe UI', 8, 'bold'), width=22, anchor='w').pack(side='left', padx=(4, 0))
            tk.Label(hdr, text="Main", bg=C_CARD, fg=C_FG_DIM,
                     font=('Segoe UI', 8, 'bold'), width=22, anchor='w').pack(side='left', padx=(4, 0))
            tk.Label(hdr, text="", bg=C_CARD, width=8).pack(side='left')

            tk.Frame(table_frame, bg=C_BORDER, height=1).pack(fill='x', pady=2)

            for fkey, flabel in compare_fields:
                d_val = dup.fields.get(fkey)
                m_val = sample.fields.get(fkey)
                d_str = str(d_val).strip() if d_val is not None and str(d_val) != 'nan' else ''
                m_str = str(m_val).strip() if m_val is not None and str(m_val) != 'nan' else ''

                if d_str == m_str:
                    status = "✓" if d_str else "—"
                    s_color = C_GREEN if d_str else C_FG_DIM
                elif not d_str:
                    status = "⚠"
                    s_color = C_AMBER
                elif not m_str:
                    status = "⚠"
                    s_color = C_AMBER
                else:
                    status = "✗"
                    s_color = C_RED

                frow = tk.Frame(table_frame, bg=C_CARD)
                frow.pack(fill='x')

                tk.Label(frow, text=flabel, bg=C_CARD, fg=C_FG_DIM,
                         font=('Segoe UI', 9), width=13, anchor='w').pack(side='left')

                d_display = (d_str[:20] + "…") if len(d_str) > 20 else (d_str or "—")
                m_display = (m_str[:20] + "…") if len(m_str) > 20 else (m_str or "—")

                d_fg = s_color if status == '✗' else C_FG
                self._selectable_label(frow, d_display, bg=C_CARD, fg=d_fg,
                                      font=('Consolas', 9), width=22
                                      ).pack(side='left', padx=(4, 0))
                self._selectable_label(frow, m_display, bg=C_CARD, fg=C_FG,
                                      font=('Consolas', 9), width=22
                                      ).pack(side='left', padx=(4, 0))
                tk.Label(frow, text=status, bg=C_CARD, fg=s_color,
                         font=('Segoe UI', 10, 'bold'), width=3).pack(side='left', padx=(4, 0))

    # ─── Finished Screen ──────────────────────────────────────────

    def _render_finished(self):
        parent = self.scroll_frame.inner
        for w in parent.winfo_children():
            w.destroy()
        self._photo_refs.clear()

        tk.Frame(parent, bg=C_BG, height=80).pack()
        tk.Label(parent, text="🎉  All matching samples reviewed!", bg=C_BG, fg=C_GREEN,
                 font=('Segoe UI', 18, 'bold')).pack()
        tk.Label(parent, text="Change filters or save your session.",
                 bg=C_BG, fg=C_FG_DIM, font=('Segoe UI', 11)).pack(pady=10)
        ttk.Button(parent, text="💾 Save Session", command=self._save_session,
                   style='Accent.TButton').pack(pady=10)
        self._update_stats()

    # ─── Image Loading ────────────────────────────────────────────

    def _load_photo(self, path: Optional[Path], max_w: int, max_h: int):
        """Load image → PhotoImage, or None."""
        if not PIL_AVAILABLE or path is None or not path.exists():
            return None

        img = None
        try:
            img = Image.open(path)
            img.load()
        except Exception:
            img = None

        if img is None and path.suffix.lower() in RAW_EXTENSIONS:
            try:
                import rawpy
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(use_camera_wb=True)
                img = Image.fromarray(rgb)
            except Exception:
                return None

        if img is None:
            return None

        if img.mode not in ('RGB', 'RGBA'):
            try:
                img = img.convert('RGB')
            except Exception:
                return None

        img.thumbnail((max_w, max_h), Image.LANCZOS)
        try:
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # ─── Session Save ─────────────────────────────────────────────

    def _save_session(self):
        if not self.data_loader:
            return
        s = self.data_loader.get_stats()
        if s['reviewed'] == 0:
            messagebox.showinfo("Nothing to save", "No samples have been reviewed yet.")
            return
        try:
            fp = self.data_loader.save_session()
            messagebox.showinfo("Saved", f"Saved {s['reviewed']} reviews to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save:\n{e}")

    # ─── Keyboard Shortcuts ───────────────────────────────────────

    def _bind_keys(self):
        def _safe(callback):
            """Only fire if no text-entry widget is focused (allows copy)."""
            def wrapper(event):
                focused = self.root.focus_get()
                if not isinstance(focused, (tk.Entry, tk.Text)):
                    callback()
            return wrapper

        self.root.bind('1', _safe(lambda: self._mark('valid')))
        self.root.bind('v', _safe(lambda: self._mark('valid')))
        self.root.bind('2', _safe(lambda: self._mark('invalid')))
        self.root.bind('i', _safe(lambda: self._mark('invalid')))
        self.root.bind('3', _safe(lambda: self._mark('uncertain')))
        self.root.bind('u', _safe(lambda: self._mark('uncertain')))
        self.root.bind('m', _safe(lambda: self.multi_fossil_var.set(not self.multi_fossil_var.get())))
        self.root.bind('<Right>', _safe(self._next_sample))
        self.root.bind('n', _safe(self._next_sample))
        self.root.bind('<space>', _safe(self._next_sample))
        self.root.bind('<Left>', _safe(self._prev_sample))
        self.root.bind('p', _safe(self._prev_sample))
        self.root.bind('<Control-s>', lambda e: self._save_session())

    # ─── Lifecycle ────────────────────────────────────────────────

    def cleanup(self):
        if self.data_loader:
            s = self.data_loader.get_stats()
            if s['reviewed'] > 0:
                ans = messagebox.askyesnocancel(
                    "Save?", f"You have {s['reviewed']} reviewed samples.\nSave before closing?")
                if ans is True:
                    self._save_session()
                elif ans is None:
                    return

    def run(self):
        self.root.mainloop()
