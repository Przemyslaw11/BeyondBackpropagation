"""Single source of figure style for the camera-ready.

Reviewer R1 called Figures 3 and 4 "completely illegible". The cause was
scaling: a 1568 px raster with 20 px glyphs placed at 0.32--0.48\\linewidth on a
122 mm LNCS text block renders type at roughly 1.7--2.5 pt. Nothing here fixes
that by enlarging fonts after the fact; instead every figure is generated at its
exact final size so ``\\includegraphics`` never rescales it, and the font sizes
below are therefore the sizes that reach the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt

#: LNCS ``\textwidth`` in millimetres (llncs.cls sets a 122 mm text block).
TEXTWIDTH_MM = 122.0

#: Base font size in points. Task 2 allows a 7 pt floor on tick labels, but the
#: acceptance criterion asks for 8 pt everywhere, so nothing here goes below it.
BASE_FONT_PT = 8.0
SMALL_FONT_PT = 8.0

#: Okabe-Ito, the eight-colour palette that survives all common CVD types.
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#767676",
}


@dataclass(frozen=True)
class SeriesStyle:
    """Colour, dash pattern and marker for one plotted series."""

    label: str
    color: str
    linestyle: str
    marker: str
    #: Six rungs of grouped bars cannot be told apart by lightness alone.
    hatch: str = ""

    def line_kwargs(self, **overrides) -> Dict:
        kwargs = {
            "color": self.color,
            "linestyle": self.linestyle,
            "marker": self.marker,
            "label": self.label,
        }
        kwargs.update(overrides)
        return kwargs


# Linestyle and marker vary with colour so every panel reads in greyscale.
SERIES: Dict[str, SeriesStyle] = {
    "bp": SeriesStyle("BP", OKABE_ITO["black"], "-", "o"),
    "bp_ds": SeriesStyle("BP-DS", OKABE_ITO["orange"], "--", "s", "///"),
    "mf_joint": SeriesStyle("MF-Joint", OKABE_ITO["sky_blue"], "-.", "^", "..."),
    "mf_recompute": SeriesStyle("MF", OKABE_ITO["bluish_green"], ":", "D", "\\\\\\"),
    "mf_cache_device": SeriesStyle(
        "MF-cache-device", OKABE_ITO["blue"], (0, (3, 1, 1, 1)), "v", "xxx"
    ),
    "mf_cache_host": SeriesStyle(
        "MF-cache-host", OKABE_ITO["reddish_purple"], (0, (5, 1)), "P", "---"
    ),
    "ff": SeriesStyle("FF", OKABE_ITO["vermillion"], "--", "s"),
    "cafo_rand": SeriesStyle("CaFo-Rand-CE", OKABE_ITO["orange"], "--", "s"),
    "cafo_dfa": SeriesStyle("CaFo-DFA-CE", OKABE_ITO["blue"], "-.", "^"),
    "mf": SeriesStyle("MF", OKABE_ITO["bluish_green"], ":", "D"),
}


def series(name: str) -> SeriesStyle:
    """Looks a series up, falling back to a neutral grey for unknown names."""
    return SERIES.get(name, SeriesStyle(name, OKABE_ITO["grey"], "-", "x"))


def figure_size(
    width_fraction: float = 1.0, aspect: float = 0.62, height_in: Optional[float] = None
) -> Tuple[float, float]:
    """Final on-page size in inches for a figure spanning ``width_fraction``.

    ``aspect`` is height/width; pass ``height_in`` to fix the height instead.
    """
    width_in = TEXTWIDTH_MM / 25.4 * width_fraction
    return (width_in, height_in if height_in is not None else width_in * aspect)


def apply_style() -> None:
    """Installs the rcParams every figure in this package is generated under."""
    mpl.rcParams.update(
        {
            # Vector output, and no raster fallback anywhere.
            "backend": "Agg",
            "pdf.compression": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.format": "pdf",
            "savefig.bbox": "standard",
            "savefig.pad_inches": 0.0,
            "savefig.transparent": False,
            # Type sizes are final sizes: figures are never rescaled by LaTeX.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": BASE_FONT_PT,
            "axes.titlesize": BASE_FONT_PT,
            "axes.labelsize": BASE_FONT_PT,
            "xtick.labelsize": SMALL_FONT_PT,
            "ytick.labelsize": SMALL_FONT_PT,
            "legend.fontsize": SMALL_FONT_PT,
            "figure.titlesize": BASE_FONT_PT,
            "mathtext.fontset": "dejavusans",
            # Thin, even rules; the defaults are far too heavy at this size.
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 0.6,
            "patch.linewidth": 0.6,
            "hatch.linewidth": 0.35,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.pad": 2.0,
            "ytick.major.pad": 2.0,
            "axes.labelpad": 2.0,
            # No in-image titles: captions live in LaTeX.
            "axes.titlepad": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.alpha": 1.0,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.borderaxespad": 0.0,
            "figure.dpi": 300,
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.01,
            "figure.constrained_layout.w_pad": 0.01,
            "figure.constrained_layout.hspace": 0.02,
            "figure.constrained_layout.wspace": 0.02,
            "axes.prop_cycle": mpl.cycler(
                color=[
                    OKABE_ITO["black"],
                    OKABE_ITO["orange"],
                    OKABE_ITO["sky_blue"],
                    OKABE_ITO["bluish_green"],
                    OKABE_ITO["blue"],
                    OKABE_ITO["vermillion"],
                    OKABE_ITO["reddish_purple"],
                ]
            ),
        }
    )


def zero_base(axis, values: Sequence[float], headroom: float = 0.08) -> None:
    """Zero-bases an axis, which is the default unless zero is meaningless."""
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return
    top = max(finite)
    axis(0.0, top * (1.0 + headroom) if top > 0 else 1.0)


def shared_legend(fig, handles, labels, ncol: int, y: float = 0.0) -> None:
    """Places one legend for the whole figure, outside every axes."""
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
    )


def save(fig, path: Path) -> Path:
    """Writes a PDF that is byte-identical across runs on identical inputs.

    The PDF backend stamps ``/CreationDate`` unless it is explicitly suppressed,
    which alone would defeat the reproducibility criterion.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    return path
