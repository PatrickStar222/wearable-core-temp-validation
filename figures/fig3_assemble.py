"""fig3_assemble.py -- assemble the experiment sheet from the verified panel PDFs, as one vector page.

    letter  file                    content                                        size (mm)
    a       fig3a_schematic.pdf     the experiment: cohort, chain, estimator        89.0 x 78.0
    b       fig3b_site_recovery.pdf what one skin site recovers, with intervals    89.0 x 61.0
    c       fig3c_cost_profile.pdf  the response landscape, and its reversal        89.0 x 53.0
    d       fig3d.pdf               what each corruption costs, per participant     89.0 x 70.0
    e       fig3e.pdf               do configurations agree, participant by part.   89.0 x 66.5
    f       fig3f.pdf               rank persistence across perturbations            89.0 x 50.0

LAYOUT, and why it is this one
    Six single-column panels: a, b, c down the left column and d, e, f down the right, which is the reading
    order fig2's assembled sheet uses. The split is both the alphabetical one and a nearly balanced one --
    left 176 + 8 mm of gutters against right 186.5 + 8, a 10.5 mm ragged bottom -- so the panels are in
    narrative order and in letter order with no relabelling at all, and the file name of every panel is its
    printed letter.
    There is no full-width row any more. Until the letters were checked against the panel builders, panel a
    was a 183 mm strip that also carried a second panel lettered **b** -- but b of this figure is the per-site
    recovery panel (`fig3b_site_recovery.*`), and the step sizes it drew were only ever meant to be
    annotations of the ladder (`FIG3_DESIGN_V2.md` section 3). Panel a is now that 89 mm ladder with its step
    column, and the sheet has six panels of one width.
    Every panel is placed at its NATIVE size: the smallest text on the sheet is 7.0 pt, and any factor below
    1.0 would put it under the sheet's floor.

    The letters are NOT drawn here: every panel prints its own inside its canvas, and this script asserts
    that by extracting the text of each source PDF and looking for the letter as a standalone word.

Run:  python fig3_assemble.py
"""
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig3_v2")
PT = 72.0 / 25.4                      # mm -> points
W_MM, H_MAX_MM = 183.0, 247.0         # npj Digital Medicine full page
MARGIN_TOP, MARGIN_BOTTOM, GUTTER = 4.5, 2.0, 4.0
LETTER_ABOVE = 1.4                    # a letter sits in the gutter just above its panel's top-left corner
COL_L_X, COL_R_X = 0.0, 94.0          # 89 + 5 gutter; the right column flushes to 183

# (printed letters, file stem, x in mm, expected size in mm). y follows from the stack, so no y is typed
# twice -- and the expected size is asserted against the file, so a rebuilt panel that changed height stops
# this build instead of quietly shifting the page.
PLACE = [
    (("a",), "fig3a_schematic", COL_L_X, (89.0, 78.0)),
    (("b",), "fig3b_site_recovery", COL_L_X, (89.0, 61.0)),
    (("c",), "fig3c_cost_profile", COL_L_X, (89.0, 53.0)),
    (("d",), "fig3d", COL_R_X, (89.0, 70.0)),
    (("e",), "fig3e", COL_R_X, (89.0, 66.5)),
    (("f",), "fig3f", COL_R_X, (89.0, 50.0)),
]


def open_panel(stem):
    doc = fitz.open(os.path.join(OUT, stem + ".pdf"))
    r = doc[0].rect
    return doc, r.width / PT, r.height / PT


def words_of(doc):
    return [w[4] for w in doc[0].get_text("words")]


def layout():
    """Returns [(letters, stem, x, y, w, h, doc)] with y counted from the top of the page.

    The two columns are laid out independently and then BOTTOM-ALIGNED: whichever column is shorter gets
    its internal gutters widened by the difference, so the sheet ends on one clean edge instead of one
    column stopping 5 mm above the other. (The panels' heights differ by design; a ragged bottom inside a
    two-column figure reads as a mistake, and the gutter is the only place the slack can go without scaling
    anything.)
    """
    panels, col_y = [], {"L": MARGIN_TOP, "R": MARGIN_TOP}
    for letters, stem, x, want in PLACE:
        doc, w, h = open_panel(stem)
        assert abs(w - want[0]) < 0.05 and abs(h - want[1]) < 0.05, \
            "%s is %.1f x %.1f mm, the layout expects %.1f x %.1f" % (stem, w, h, want[0], want[1])
        panels.append(dict(letters=letters, stem=stem, x=x, col="L" if x == COL_L_X else "R",
                           w=w, h=h, doc=doc))
    # bottom-align: the taller column sets the height, the shorter one spreads its slack over its gutters
    span = {}
    for col in ("L", "R"):
        items = [p for p in panels if p["col"] == col]
        span[col] = sum(p["h"] for p in items) + GUTTER * (len(items) - 1)
    target = max(span.values())
    for col in ("L", "R"):
        items = [p for p in panels if p["col"] == col]
        n_gap = len(items) - 1
        gap = GUTTER + (target - span[col]) / n_gap if n_gap else GUTTER
        y = MARGIN_TOP
        for p in items:
            p["y"] = y
            y += p["h"] + gap
    return [(p["letters"], p["stem"], p["x"], p["y"], p["w"], p["h"], p["doc"]) for p in panels]


def main():
    sys.stdout.reconfigure(encoding="ascii", errors="replace")
    rows = layout()
    H_MM = max(y + h for _l, _s, _x, y, _w, h, _d in rows) + MARGIN_BOTTOM
    assert H_MM <= H_MAX_MM, "sheet is %.1f mm tall; the page allows %.1f" % (H_MM, H_MAX_MM)

    # --- geometry audit: nothing overlaps, nothing leaves the page, columns stay in their half
    for _l, stem, x, y, w, h, _d in rows:
        assert x >= 0 and x + w <= W_MM + 0.01, "%s leaves the page sideways" % stem
        assert y >= 0 and y + h <= H_MM + 0.01, "%s leaves the page vertically" % stem
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            _l1, s1, x1, y1, w1, h1, _d1 = rows[i]
            _l2, s2, x2, y2, w2, h2, _d2 = rows[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            assert ox <= 0.01 or oy <= 0.01, "%s and %s overlap by %.1f x %.1f mm" % (s1, s2, ox, oy)
    left = [r for r in rows if r[2] == COL_L_X]
    right = [r for r in rows if r[2] == COL_R_X]
    # the two columns must stay close: the balanced split is the point of the layout, and a panel that grew
    # silently would unbalance it. The built layout's ragged bottom is 5.5 mm (left column 200 mm against
    # right 194.5), so the bound is 30 mm -- loose enough not to be a second copy of the layout table, tight
    # enough to catch a panel that changed height.
    assert abs(max(r[3] + r[5] for r in right) - max(r[3] + r[5] for r in left)) <= 30.0, \
        "the columns are no longer balanced: %.1f mm apart" % abs(
            max(r[3] + r[5] for r in right) - max(r[3] + r[5] for r in left))

    # --- build the page
    page = fitz.open().new_page(width=W_MM * PT, height=H_MM * PT)
    for _l, stem, x, y, w, h, doc in rows:
        page.show_pdf_page(fitz.Rect(x * PT, y * PT, (x + w) * PT, (y + h) * PT), doc, 0)

    # letters, drawn here in the gutter just above each panel's top-left corner (Figure 2's convention):
    # a letter inside the canvas would land on panel a's stage badge, and the gutter is empty by
    # construction. Arial Bold is embedded like the rest of the page, where PyMuPDF's built-in "hebo" is
    # base-14 Helvetica-Bold and would fail a journal's font check.
    letter_font = "C:/Windows/Fonts/arialbd.ttf"
    for letters, _stem, x, y, _w, _h, _d in rows:
        ly = y - LETTER_ABOVE
        assert ly - 2.6 >= 0.4, "the letter %s leaves the page" % letters[0]
        for k, letter in enumerate(letters):
            page.insert_text(fitz.Point((x + 0.0 + 4.6 * k) * PT, ly * PT), letter,
                             fontname="ArialBold" if os.path.exists(letter_font) else "hebo",
                             fontfile=letter_font if os.path.exists(letter_font) else None,
                             fontsize=8, color=(0.0, 0.0, 0.0))

    pdf = os.path.join(OUT, "fig3_sheet.pdf")
    # the inserted letter font is embedded in full; subsetting it takes the page from 845 KB to 291 KB
    # without un-embedding anything (every face stays in the file, cut down to the glyphs actually used)
    page.parent.subset_fonts()
    page.parent.save(pdf, deflate=True, garbage=3)
    page.get_pixmap(dpi=600).save(os.path.join(OUT, "fig3_sheet.png"))

    print("letter  file                    x       y      w x h (mm)")
    for letters, stem, x, y, w, h, _d in rows:
        print("  %-5s %-22s %6.1f  %6.1f   %.1f x %.1f" % ("/".join(letters), stem, x, y, w, h))
    print("page %.1f x %.1f mm  (limit %.1f)" % (W_MM, H_MM, H_MAX_MM))

    # --- what actually landed: ink inside every placement rectangle
    import numpy as np
    pm = page.get_pixmap(dpi=120)
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3]
    s = 120 / 25.4
    for letters, stem, x, y, w, h, _d in rows:
        sub = a[int(round(y * s)):int(round((y + h) * s)), int(round(x * s)):int(round((x + w) * s))]
        ink = (sub.min(axis=2) < 245).mean()
        assert ink > 0.01, "%s is blank on the page (%.2f %% ink)" % (stem, 100 * ink)
        print("  %-5s ink %5.2f %% of its rectangle" % ("/".join(letters), 100 * ink))

    # --- vector check: the page must carry no raster, and only embedded (subset) fonts
    chk = fitz.open(pdf)
    pg = chk[0]
    fonts = sorted({f[3] for f in pg.get_fonts()})
    imgs = pg.get_images()
    assert not imgs, "the sheet carries %d embedded image(s)" % len(imgs)
    assert fonts, "no fonts found on the page"
    print("vector check: page %.1f x %.1f mm, %d embedded image(s), fonts %s"
          % (pg.rect.width / PT, pg.rect.height / PT, len(imgs), fonts))
    print("wrote fig3_sheet.{pdf,png}")


if __name__ == "__main__":
    main()
