r"""Build the two main-text data tables as a formatted Word file.

House style: Times New Roman, three-line (booktabs) rules, no vertical lines, bold header, numeric
columns right-aligned, caption above the table, footnote below. Delta is set as inline OMML so it
renders as the italic math capital, not the upright Unicode capital.

Table 1  cross-site recovery          11 sites, nRMSE with a bootstrap interval
Table 2  drift attribution            three configurations, four numeric columns
"""
import os

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

import _omml as O

OUT = 'Results_tables.docx'
W_URI = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

# ---------------------------------------------------------------- data (all verified)
SITES = [
    ('head', 0.9706, 0.9545, 0.9898),
    ('upper back', 0.9676, 0.9492, 0.9908),
    ('lower back', 0.9331, 0.9052, 0.9618),
    ('hand', 0.9199, 0.9026, 0.9511),
    ('chest', 0.9105, 0.8767, 0.9544),
    ('lower arm', 0.8988, 0.8674, 0.9349),
    ('abdomen', 0.8966, 0.8538, 0.9456),
    ('upper arm', 0.8430, 0.8056, 0.8852),
    ('calf', 0.8173, 0.7876, 0.8502),
    ('thigh', 0.8041, 0.7746, 0.8453),
    ('foot', 0.7132, 0.6894, 0.7451),
]

DRIFT = [
    ('B3 eleven skin sites', 0.6899, 0.6690, -0.02844, 98, 34),
    ('B4 skin + heart rate', 0.6659, 0.6546, -0.01486, 92, 40),
    ('B5 skin + heart rate + environment', 0.6381, 0.6229, -0.01923, 95, 37),
]


def set_cell_margins(cell, top=60, bottom=60, left=110, right=110):
    tcPr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement('w:tcMar')
    for tag, val in (('top', top), ('start', left), ('bottom', bottom), ('end', right)):
        el = OxmlElement('w:%s' % tag)
        el.set(qn('w:w'), str(val))
        el.set(qn('w:type'), 'dxa')
        mar.append(el)
    tcPr.append(mar)


def cell_borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement('w:tcBorders')
    for side, spec in (('top', top), ('bottom', bottom)):
        el = OxmlElement('w:%s' % side)
        if spec is None:
            el.set(qn('w:val'), 'nil')
        else:
            el.set(qn('w:val'), 'single')
            el.set(qn('w:sz'), str(spec))
            el.set(qn('w:space'), '0')
            el.set(qn('w:color'), '000000')
        borders.append(el)
    for side in ('start', 'end', 'left', 'right'):
        el = OxmlElement('w:%s' % side)
        el.set(qn('w:val'), 'nil')
        borders.append(el)
    tcPr.append(borders)


def normalise(parts):
    """Accept a plain string, or a list of ('t', text) / ('m', latex) pairs."""
    if isinstance(parts, str):
        return [('t', parts)]
    out = []
    for item in parts:
        if isinstance(item, str):
            out.append(('t', item))
        elif len(item) == 2 and item[0] in ('t', 'm'):
            out.append(item)
        else:
            raise ValueError('bad cell chunk: %r' % (item,))
    return out


def write_cell(cell, parts, size=9.5, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, indent=None):
    """parts: a string, or a list of ('t', text) / ('m', latex) chunks."""
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(1.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.0
    p.alignment = align
    if indent is not None:
        p.paragraph_format.left_indent = indent
    for kind, payload in normalise(parts):
        if kind == 't':
            r = p.add_run(payload)
            r.font.size = Pt(size)
            r.font.bold = bold
        else:
            p._p.append(parse_xml(O.to_omml(payload, display=False)))
    return p


def caption(doc, text_parts, size=10):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    for kind, payload in text_parts:
        if kind == 't':
            r = p.add_run(payload)
            r.font.size = Pt(size)
            r.font.bold = True
        else:
            p._p.append(parse_xml(O.to_omml(payload, display=False)))
    return p


def footnote(doc, text, size=8.5):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


def build_table(doc, header, rows, widths, aligns, sizes=9.5):
    t = doc.add_table(rows=len(rows) + 1, cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for j, h in enumerate(header):
        c = t.cell(0, j)
        write_cell(c, h, size=sizes, bold=True, align=aligns[j])
        set_cell_margins(c)
        cell_borders(c, top=12, bottom=6)
    for i, row in enumerate(rows, start=1):
        last = (i == len(rows))
        for j, val in enumerate(row):
            c = t.cell(i, j)
            write_cell(c, val, size=sizes, align=aligns[j])
            set_cell_margins(c)
            cell_borders(c, bottom=12 if last else None)
    for j, w in enumerate(widths):
        for i in range(len(rows) + 1):
            t.cell(i, j).width = w
    return t


def main():
    d = docx.Document()
    st = d.styles['Normal']
    st.font.name = 'Times New Roman'
    st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    for s in d.sections:
        s.left_margin = s.right_margin = Cm(2.2)

    # ---------------------------------------------------------------- Table 1
    caption(d, [('t', 'Table 1 | '),
                ('t', 'Core-temperature recovery from a single skin site. '),
                ('t', 'Each of the eleven skin sites used alone, on the same 7,534 '
                      'complete-case rows, with the same estimator and validation.')])
    rows = [[s[0], '%.4f' % s[1], '%.4f\u2013%.4f' % (s[2], s[3])] for s in SITES]
    build_table(
        d,
        ['Site', 'nRMSE', '95 % CI'],
        rows,
        widths=[Cm(4.0), Cm(2.6), Cm(3.4)],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
    )
    footnote(d, 'Site is the anatomical location of the single channel supplied to the estimator. '
                'Intervals are bootstrap 95 % confidence intervals over participants. The same '
                'estimator, preprocessing pipeline and leave-one-subject-out validation are used '
                'for every row, so the only quantity that differs between rows is which channel '
                'is read. Source: fig3_v2/_panel_b_sites.csv, written by _panel_b_stats.py.')

    # ---------------------------------------------------------------- Table 2
    caption(d, [('t', 'Table 2 | '),
                ('t', 'Attribution of the drift-induced change in pooled normalised error. '),
                ('t', 'A within-record ramp from 0 to +1.0 \u00b0C added to all eleven skin '
                      'channels. The same 132 units and 7,534 rows underlie all three '
                      'configurations, so they differ in sensitivity to one perturbation rather '
                      'than in coverage.')])
    rows = []
    for name, clean, drifted, dsq, better, worse in DRIFT:
        rows.append([
            name,
            '%.4f \u2192 %.4f' % (clean, drifted),
            '%+.5f' % dsq,
            '%d / %d' % (better, worse),
        ])
    build_table(
        d,
        ['Configuration', 'nRMSE, clean \u2192 drift',
         [('m', r'\Delta'), ('t', ' squared deviation')], 'Units better / worse'],
        rows,
        widths=[Cm(5.0), Cm(3.4), Cm(3.2), Cm(3.0)],
        aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT,
                WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT],
    )
    footnote(d, 'The change in squared deviation is negative because a lower nRMSE is better. It is '
                'computed on squared deviations and is therefore larger than the change in '
                'normalised error itself. Its decomposition follows Eq. (6): the reweighting and '
                'interaction terms are both exactly zero for this perturbation, so the whole change '
                'is within-unit. Repeating the drift with the sign of the ramp drawn at random '
                'within each unit at unchanged magnitude reverses the improvement into a loss in '
                'all three configurations (+1.74 %, +0.51 % and +1.33 % on average over ten draws, '
                'with 0, 1 and 0 of those draws improving). Units better and units worse count the '
                'individual units whose error fell and rose. Source: _drift_decomp_summary.csv, '
                'written by _drift_decomp.py.')

    d.save(OUT)
    print('saved %s (%d bytes)' % (OUT, os.path.getsize(OUT)))
    print('tables: %d | paragraphs: %d' % (len(d.tables), len(d.paragraphs)))


if __name__ == '__main__':
    main()
