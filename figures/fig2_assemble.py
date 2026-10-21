"""Assemble Figure 2 from the verified panel PDFs, as one vector page.

Every panel is placed at its NATIVE size: none of these figures is scaled, because the smallest text
in them is 5.0-5.2 pt and any scale factor under ~0.95 would push it below the 5 pt floor.

    panel letters   file              content                                        dataset
    a               s1_comic2.pdf     the four datasets, one column each             key
    b               body_sites.pdf    the eleven skin sites on the body              [1]
    c               panel_b.pdf       11 sites across nine ambient conditions        [1]
    d               panel_d.pdf       ear-canal vs esophageal transfer functions     [2]
    e               panel_a.pdf       skin, core and heart rate over 180 min         [3]
    f               panel_e.pdf       24-h personal air and skin, urban vs rural     [4]

Layout: the dataset key is 183 mm wide and can only be a full-width row; it goes first, then two
columns of 89 mm with a 4 mm gutter. a/b/c run down the left column, d/e/f down the right, which
keeps the panels in dataset order.

Run:  python fig2_assemble.py
"""
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig2_v2")
PT = 72.0 / 25.4                      # mm -> points
INK = (0.29, 0.25, 0.20)              # #4A4033, the ink used by every panel
# The five data panels are plain vector charts, so the composite uses the SMOOTH edition of the
# schematic strip by default; the hand-drawn strip stays available with --hand.
HAND = "--hand" in sys.argv
KEY = "s1_comic2" if HAND else "s1_comic2_smooth"
STEM_OUT = "fig2_assembled" if not HAND else "fig2_assembled_hand"
# Letters must be an EMBEDDED font: PyMuPDF's built-in "hebo" is base-14 Helvetica-Bold, which is
# not embedded and would fail a journal's font check. Arial Bold matches the panels' own Arial and
# gets subset-embedded like the rest of the page.
LETTER_FONT = "C:/Windows/Fonts/arialbd.ttf"
LETTER_NAME = "ArialBold"
W_MM, H_MAX_MM = 183.0, 247.0         # Nature full page
MARGIN_TOP, MARGIN_BOTTOM, GUTTER = 4.5, 2.0, 4.0
COL_R_X = 94.0                        # right column starts here; its panels flush to 183

# (letter, file stem, x in mm from the left, y in mm from the top). y is computed in main() so the
# whole stack follows from the panel heights rather than from numbers typed twice.
PLACE = [
    ("a", KEY, 0.0),
    ("b", "body_sites", 0.0),
    ("c", "panel_b", 0.0),
    ("d", "panel_d", None),           # right column, right-aligned
    ("e", "panel_a", None),
    ("f", "panel_e", None),
]


def size_of(stem):
    d = fitz.open(os.path.join(OUT, stem + ".pdf"))
    r = d[0].rect
    return d, r.width / PT, r.height / PT


def main():
    sys.stdout.reconfigure(encoding="ascii", errors="replace")
    docs, rects, letters = {}, [], []

    # the key strip is a full-width row of its own; the two columns start under it
    letter, stem, _x = PLACE[0]
    doc, w, h = size_of(stem)
    docs[stem] = doc
    rects.append((letter, stem, 0.0, MARGIN_TOP, w, h))
    letters.append((letter, 0.0, MARGIN_TOP - 1.2))
    col_y = {"L": MARGIN_TOP + h + GUTTER, "R": MARGIN_TOP + h + GUTTER}

    for letter, stem, x in PLACE[1:]:
        doc, w, h = size_of(stem)
        docs[stem] = doc
        col = "L" if x == 0.0 else "R"
        y = col_y[col]
        col_y[col] = y + h + GUTTER
        x0 = x if x is not None else W_MM - w
        rects.append((letter, stem, x0, y, w, h))
        letters.append((letter, x0, y - 1.2))

    H_MM = max(y + h for _l, _s, _x, y, _w, h in rects) + MARGIN_BOTTOM
    assert H_MM <= H_MAX_MM, "figure is %.1f mm tall; the page allows %.1f" % (H_MM, H_MAX_MM)

    # --- geometry audit: nothing overlaps, nothing leaves the page, letters sit in the gutters
    for letter, stem, x, y, w, h in rects:
        assert 0 <= x and x + w <= W_MM + 0.01, "%s leaves the page sideways" % stem
        assert 0 <= y and y + h <= H_MM + 0.01, "%s leaves the page vertically" % stem
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            _l1, s1, x1, y1, w1, h1 = rects[i]
            _l2, s2, x2, y2, w2, h2 = rects[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            assert ox <= 0.01 or oy <= 0.01, "%s and %s overlap by %.1f x %.1f mm" % (s1, s2, ox, oy)
    for letter, lx, ly in letters:
        lw, lh = 1.9, 2.4                                  # an 8 pt bold lowercase letter
        for _l, stem, x, y, w, h in rects:
            ox = min(lx + lw, x + w) - max(lx, x)
            oy = min(ly, y + h) - max(ly - lh, y)
            assert ox <= 0.05 or oy <= 0.05, "letter %s sits on %s" % (letter, stem)
        assert ly - lh >= 0, "letter %s leaves the page" % letter

    # --- build the page
    page = fitz.open().new_page(width=W_MM * PT, height=H_MM * PT)
    for _letter, stem, x, y, w, h in rects:
        page.show_pdf_page(fitz.Rect(x * PT, y * PT, (x + w) * PT, (y + h) * PT), docs[stem], 0)
    for letter, lx, ly in letters:
        if os.path.exists(LETTER_FONT):
            page.insert_text(fitz.Point(lx * PT + 1.0, ly * PT), letter, fontname=LETTER_NAME,
                             fontfile=LETTER_FONT, fontsize=8, color=INK)
        else:
            page.insert_text(fitz.Point(lx * PT + 1.0, ly * PT), letter, fontname="hebo",
                             fontsize=8, color=INK)

    pdf = os.path.join(OUT, STEM_OUT + ".pdf")
    page.parent.save(pdf, deflate=True, garbage=3)
    page.get_pixmap(dpi=600).save(os.path.join(OUT, STEM_OUT + ".png"))

    print("panel  file          x       y       w x h (mm)")
    for letter, stem, x, y, w, h in rects:
        print("  %s    %-12s %6.1f  %6.1f   %.1f x %.1f" % (letter, stem, x, y, w, h))
    print("page %.1f x %.1f mm  (limit %.1f)" % (W_MM, H_MM, H_MAX_MM))

    # --- what actually landed: ink inside every placement rectangle
    pm = page.get_pixmap(dpi=120)
    import numpy as np
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, pm.n)[:, :, :3]
    s = 120 / 25.4
    for letter, stem, x, y, w, h in rects:
        sub = a[int(y * s):int((y + h) * s), int(x * s):int((x + w) * s)]
        ink = (sub.min(axis=2) < 245).mean()
        assert ink > 0.02, "%s is blank on the page (%.1f %% ink)" % (stem, 100 * ink)
        print("  %s ink %5.1f %% of its rectangle" % (letter, 100 * ink))
    chk = fitz.open(pdf)
    pg = chk[0]
    fonts = {f[3] for f in pg.get_fonts()}
    imgs = pg.get_images()
    print("vector check: page %.1f x %.1f mm, %d embedded image(s), fonts %s"
          % (pg.rect.width / PT, pg.rect.height / PT, len(imgs), sorted(fonts)))
    print("wrote %s.{pdf,png}" % STEM_OUT)


if __name__ == "__main__":
    main()
