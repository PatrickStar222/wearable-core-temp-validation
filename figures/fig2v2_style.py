"""
fig2v2_style.py -- design tokens for the Figure 2 rebuild (v2).

This module is the single source of truth for colour, type and stroke weight.
Panels import it; they do not carry literals. Two decisions are worth recording
because the previous build got both wrong.

1. TYPE. The v1 build used DejaVu Sans everywhere because Arial lacks U+207B
   (superscript minus). That is true but it is not a reason to abandon Arial:
   superscripts belong in mathtext, not in a literal glyph. `_font_probe.py`
   measured 18 families and `_font_mathtext_check.py` proved that Arial with
   `mathtext.fontset='custom'` renders 10^-3, degC degC^-1, delta and rho with no
   tofu. So the figure is Arial, and `sub()` / `sup()` below are the only
   sanctioned ways to set a super/subscript.

2. COLOUR. v1 leant on near-white tints for the environmental ramp, which reads
   grey in print. v2 makes the ambient family a six-step amber LIGHTNESS ladder
   that is used at large area, so the page is warm rather than washed out. The
   ladder is ordered by lightness, not only hue, which is what keeps it legible
   under protan/deutan/tritan simulation and in greyscale.

Every colour role carries a meaning, fixed for the whole figure:

    AMB   environmental forcing        warm amber, six-step ladder
    SKIN  peripheral observation       coral; opacity encodes site sensitivity
    CORE  core reference / the truth   deep teal
    COH   replication cohorts          moss green
    EAR   ear-canal ambient ladder     indigo, three-step ladder
    DISP  dispersion as a 2nd channel  slate indigo
    neutral ink / grey / hair / paper
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

# --------------------------------------------------------------- palette
# Environmental forcing: one hue, six lightness steps. Placed on an ink-like
# curve from a warm cream to a bronze, so a nine-condition strip can be read as
# "colder -> warmer" without a legend. The mid steps are deliberately VIVID --
# an earlier draft used a beige ramp and the whole page read as washed out, which
# is the single defect this ladder exists to avoid.
AMB_1 = "#F7E7B8"
AMB_2 = "#F0D283"
AMB_3 = "#E5B44B"
AMB_4 = "#D29419"
AMB_5 = "#B0740C"
AMB_6 = "#8A5906"
AMB = AMB_5
AMB_TXT = "#7A5008"
AMB_LADDER = [AMB_1, AMB_2, AMB_3, AMB_4, AMB_5, AMB_6]

# Peripheral observation. Coral is the Okabe-Ito orange axis, so it separates
# from the teal core under every simulated colour-vision deficiency.
SKIN = "#D9503A"
SKIN_D = "#AE3620"
SKIN_XL = "#F7D9D2"

# Core reference. The only deep cool colour, so it always reads as "the truth".
# Kept at L* ~36: a mid teal (L* 50) collapses onto the coral under deuteranopia.
CORE = "#0E5F7A"
CORE_D = "#0A4557"
CORE_L = "#D3E7ED"

# Replication cohorts. Richer than the earlier #79A65A so it does not read grey
# beside the amber.
COH = "#5E9B3E"
COH_D = "#3F6E27"

# Ear-canal ambient ladder: measured span 45 L*, well past the ~12 L* a reader
# needs between adjacent ordinal steps.
EAR_1 = "#C2C6E4"
EAR_2 = "#7B82B8"
EAR_3 = "#3A3F76"
EAR_LADDER = [EAR_1, EAR_2, EAR_3]

# Dispersion, as an independent second encoding in panel e.
DISP = "#5F66A3"

# Neutrals. Deliberately blue-leaning: a neutral grey next to warm amber reads
# as dirty, a blue-leaning ink reads as intentional.
INK = "#1F2A37"
SUB = "#3E4C5A"
GREY = "#5D6B76"
HAIR = "#D3DAE0"
PAPER = "#FFFFFF"

# Sequential map built from the ambient ladder; used for the condition strip and
# for any continuous ambient scale. Single-hue, no rainbow.
AMB_CMAP = LinearSegmentedColormap.from_list(
    "fig2v2_amb", [AMB_1, AMB_2, AMB_3, AMB_4, AMB_5, AMB_6])

# ------------------------------------------------------------------- type
# Three levels plus the floors, set for a 183 mm double-column page reproduced
# at 1:1. 6.0 pt is the floor; nothing in the delivered figure is smaller.
T = dict(
    # Nature's figure specification is explicit: "Separate panels in multi-panelled
    # figures should be labelled with 8-pt bold, upright (not italic) and lowercase a,
    # b, c, etc." The token was 10.0 pt from the v2 build; 8.0 is the house value.
    letter=8.0,    # panel letter, bold
    title=7.6,     # sentence-case panel title, bold
    sub=6.2,       # provenance / subtitle line
    ax=7.2,        # axis label
    tick=6.4,      # tick label
    val=6.8,       # in-plot numerical annotation, bold
    note=6.0,      # smallest permitted annotation
    lg=6.6,        # legend
)
FONT = "Arial"
FALLBACK = ["Arial", "Helvetica", "DejaVu Sans"]

# Stroke weights, points.
LW = dict(axis=0.6, series=2.0, hero=2.2, thin=0.5, guide=0.6, mean=2.4,
          arrow=1.5, band=0.9)
# Marker sizes, points.
MS = dict(dot=2.6, med=3.4, big=4.6, mean=5.2)


def apply() -> None:
    """Install the token set into rcParams. Call once, before any figure exists."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FALLBACK,
        "font.size": T["tick"],
        "axes.labelsize": T["ax"],
        "axes.titlesize": T["title"],
        "xtick.labelsize": T["tick"],
        "ytick.labelsize": T["tick"],
        "legend.fontsize": T["lg"],
        "axes.linewidth": LW["axis"],
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": LW["axis"],
        "ytick.major.width": LW["axis"],
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": LW["series"],
        "lines.solid_capstyle": "round",
        "figure.dpi": 300,
        "savefig.dpi": 600,
        # No tight bounding box: a submission figure must be exactly the size the
        # layout says. With bbox="tight" the delivered PNG changes dimensions
        # with its content, which silently moves every panel between builds.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        # Superscripts go through mathtext with Arial faces, which is how the
        # string "10^-3" renders without the U+207B glyph Arial does not have.
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": FONT + ":italic",
        "mathtext.bf": FONT + ":bold",
        "mathtext.default": "regular",
    })


# --------------------------------------------------------------- helpers
def tint(color: str, f: float) -> str:
    """Flat tint toward white. No alpha, so PDF output stays flat and print-safe."""
    r, g, b = to_rgb(color)
    return to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def shade(color: str, f: float) -> str:
    """Flat shade toward black."""
    r, g, b = to_rgb(color)
    return to_hex((r * (1 - f), g * (1 - f), b * (1 - f)))


def sup(s: str) -> str:
    """Superscript via mathtext -- the only sanctioned route (Arial has no U+207B)."""
    return "$^{" + s + "}$"


def sub(s: str) -> str:
    """Subscript via mathtext."""
    return "$_{" + s + "}$"


def degc() -> str:
    """The degree-Celsius unit, set the way the journal wants it.

    NOT "$^\\circ$C": the \\circ glyph is absent from Arial's mathtext set, so matplotlib
    silently falls back to STIXGeneral-Regular -- a Times face -- and the PDF then ships with a
    font Nature's Arial/Helvetica rule does not allow. It happened to panels d and e, which both
    embedded STIXGeneral-Regular until _probe_degree.py caught it. "$\\degree$C" also stays in
    Arial, but the literal glyph is the plainer spelling. Run that probe before changing this.
    """
    return "(\N{DEGREE SIGN}C)"


def lstar(color: str) -> float:
    """CIE L* for a hex colour. Used by the token audit, not by the panels."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(v) for v in to_rgb(color))
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 116 * (y ** (1 / 3)) - 16 if y > 0.008856 else 903.3 * y


def contrast(fg: str, bg: str) -> float:
    """WCAG contrast ratio, for checking annotation legibility on tinted bands."""
    def rel(c: str) -> float:
        def lin(v: float) -> float:
            return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (lin(v) for v in to_rgb(c))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = rel(fg), rel(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def tidy(ax) -> None:
    """Left and bottom spines only, with journal tick padding."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(pad=2, width=LW["axis"], length=2.4)


def panel_letter(fig, ax, letter: str, dx_mm: float = 0.0, dy_mm: float = 0.0):
    """Bold lowercase panel letter at the top-left, the Nature convention."""
    bb = ax.get_position()
    fig.text(bb.x0 + dx_mm / (fig.get_figwidth() * 25.4),
             bb.y1 + dy_mm / (fig.get_figheight() * 25.4),
             letter, fontsize=T["letter"], fontweight="bold", color=INK,
             ha="left", va="baseline")


if __name__ == "__main__":
    apply()
    print("palette roles and measured lightness")
    for name, col in [("AMB_1", AMB_1), ("AMB_2", AMB_2), ("AMB_3", AMB_3),
                      ("AMB_4", AMB_4), ("AMB_5", AMB_5), ("AMB_6", AMB_6),
                      ("SKIN", SKIN), ("CORE", CORE), ("COH", COH),
                      ("EAR_1", EAR_1), ("EAR_2", EAR_2), ("EAR_3", EAR_3),
                      ("DISP", DISP), ("INK", INK), ("GREY", GREY),
                      ("HAIR", HAIR)]:
        print("  %-7s %-9s L* %5.1f  contrast-on-white %5.2f"
              % (name, col, lstar(col), contrast(col, PAPER)))
    print("\nambient ladder L* steps:", [round(lstar(c), 1) for c in AMB_LADDER])
    print("ear ladder L* steps:    ", [round(lstar(c), 1) for c in EAR_LADDER])
    print("amber text on amber-1 band contrast: %.2f" % contrast(AMB_TXT, AMB_1))
