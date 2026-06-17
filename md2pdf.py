"""Convert deepArticle.md to PDF with proper math rendering via matplotlib mathtext.

Uses: fpdf2, Pillow, matplotlib (all trusted PyPI packages). No pandoc/LaTeX needed.
"""

import re, io, hashlib
from pathlib import Path
from fpdf import FPDF
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_FILE = Path(__file__).parent / "deepArticle.md"
OUTPUT_FILE = Path(__file__).parent / "deepArticle.pdf"
CHART_DIR = Path(__file__).parent
CACHE_DIR = Path(__file__).parent / ".math_cache"
CACHE_DIR.mkdir(exist_ok=True)

CHARTS = [
    ("hayward_results.png", "qnm shifts and echo thresholds",
     "Hayward black hole: QNM frequency shifts and relative echo amplitude vs core radius L/M."),
    ("tov_consistency_plot.png", "tov_selfconsistent_check",
     "TOV self-consistency check: Hayward exact density vs free ansatz, best-fit comparison."),
    ("theta_dynamics_results.png", "superluminal geodesic angle dynamics",
     "1+1D superluminal geodesic angle dynamics: phase as function of impact parameter and energy."),
    ("theta_1p3d_results.png", "theta_1p3d_phase",
     "1+3D group non-closure: departure from identity for random superluminal path pairs."),
]

# ═══════════════════════════════════════════════════════════════
# Math renderer
# ═══════════════════════════════════════════════════════════════

class MathRenderer:
    """Renders LaTeX math as PNG via matplotlib mathtext (no LaTeX install needed)."""

    def __init__(self, dpi=200):
        self.dpi = dpi

    @staticmethod
    def sanitize(expr):
        s = expr.strip()
        s = re.sub(r'\\text\{([^}]*)\}', r'\\mathrm{\1}', s)
        s = re.sub(r'\\mathrm\{([^}]*)\}', r'\\mathrm{\1}', s)  # already handled
        s = s.replace('\\substack', '\\scriptstyle')
        # Unsupported symbols → remove or replace
        s = s.replace('\\Box', '\\square')  # neither supported, will fallback
        s = s.replace('\\square', '\\Box')
        # Better: use Unicode for \Box / d'Alembertian
        s = s.replace('\\Box', '\u25a1')  # □ white square
        # Strip \left \right size modifiers — mathtext auto-sizes delimiters
        # Be careful not to match \leftarrow, \rightarrow etc.
        s = re.sub(r'\\left\s*([\[\(\{|\.\\])', r'\1', s)
        s = re.sub(r'\\right\s*([\]\)\}\|\.\\])', r'\1', s)
        s = re.sub(r'\\boxed\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\big\s*\|', r'|', s)
        s = re.sub(r'\\Big\s*\|', r'|', s)
        s = re.sub(r'\\bigg\s*\|', r'|', s)
        s = re.sub(r'\\Bigg\s*\|', r'|', s)
        s = re.sub(r'\\DeclareMathOperator\*?\{[^}]*\}\{[^}]*\}', '', s)
        s = re.sub(r'\\xrightarrow\{[^}]*\}', r'\\rightarrow', s)
        s = re.sub(r'\\xleftarrow\{[^}]*\}', r'\\leftarrow', s)
        s = re.sub(r'\\overset\{[^}]*\}\{([^}]*)\}', r'\1', s)
        s = re.sub(r'\\underset\{[^}]*\}\{([^}]*)\}', r'\1', s)
        # Fix unmatched \left... with missing \right (mathtext is strict)
        s = re.sub(r'\\left\[([^\\]*)$', r'[\1', s)
        s = re.sub(r'\\left\(([^\\]*)$', r'(\1', s)
        return s

    def render(self, expr, fontsize=12, display=False):
        key = hashlib.md5(f"{expr}|{fontsize}|{display}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{key}.png"
        if cache_path.exists():
            img = Image.open(cache_path)
            return img, img.width / self.dpi * 25.4, img.height / self.dpi * 25.4

        safe = self.sanitize(expr)
        try:
            return self._render_impl(safe, fontsize, display, cache_path)
        except (ValueError, RuntimeError) as e:
            # Fallback: render as monospace text image
            print(f"  [math fallback] {expr[:60]}... -> {e}")
            return self._render_text_fallback(expr, fontsize, display, cache_path)

    def _render_impl(self, safe, fontsize, display, cache_path):
        """Actual mathtext rendering."""
        # Measure first
        fig, ax = plt.subplots(figsize=(0.01, 0.01), dpi=self.dpi)
        if display:
            text = ax.text(0.5, 0.5, f"${safe}$", fontsize=fontsize, ha='center', va='center')
        else:
            text = ax.text(0.0, 0.0, f"${safe}$", fontsize=fontsize, ha='left', va='baseline')
        ax.axis('off')
        fig.canvas.draw()
        bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox = bbox.expanded(1.15, 1.3)
        bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
        plt.close(fig)

        w_in, h_in = max(bbox_inches.width, 0.3), bbox_inches.height
        fig = plt.figure(figsize=(w_in, h_in), dpi=self.dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        if display:
            ax.text(0.5, 0.5, f"${safe}$", fontsize=fontsize, ha='center', va='center')
        else:
            ax.text(0.03, 0.10, f"${safe}$", fontsize=fontsize, ha='left', va='baseline')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                    pad_inches=0.04, transparent=True)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.save(cache_path)
        buf.close()
        return img, img.width / self.dpi * 25.4, img.height / self.dpi * 25.4

    def _render_text_fallback(self, expr, fontsize, display, cache_path):
        """Fallback: render math as monospace LaTeX text."""
        w_in = max(len(expr) * 0.08, 0.5)
        h_in = 0.3 if not display else 0.5
        fig = plt.figure(figsize=(w_in, h_in), dpi=self.dpi, facecolor='#f5f5f5')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.text(0.02, 0.15, expr, fontsize=7, family='monospace', color='#555555',
                ha='left', va='baseline', wrap=True)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                    pad_inches=0.06, facecolor='#f5f5f5')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.save(cache_path)
        buf.close()
        return img, img.width / self.dpi * 25.4, img.height / self.dpi * 25.4


# ═══════════════════════════════════════════════════════════════
# PDF
# ═══════════════════════════════════════════════════════════════

class ArticlePDF(FPDF):
    def __init__(self, mr):
        super().__init__('P', 'mm', 'A4')
        self.set_auto_page_break(True, 20)
        self.add_font("Body", "", "C:/Windows/Fonts/arial.ttf")
        self.add_font("Body", "B", "C:/Windows/Fonts/arialbd.ttf")
        self.add_font("Body", "I", "C:/Windows/Fonts/ariali.ttf")
        self.add_font("Body", "BI", "C:/Windows/Fonts/arialbi.ttf")
        self.add_font("Mono", "", "C:/Windows/Fonts/consola.ttf")
        self.mr = mr
        self.charts_done = set()

    def header(self):
        if self.page_no() <= 1:
            return
        self.set_font("Body", "I", 7)
        self.set_text_color(120)
        self.cell(0, 4, "Black Holes, GWs, and Superluminal Observers", align="C")
        self.ln(5)

    def footer(self):
        if self.page_no() <= 1:
            return
        self.set_y(-15)
        self.set_font("Body", "I", 7)
        self.set_text_color(120)
        self.cell(0, 10, str(self.page_no()), align="C")

    def try_chart(self, text):
        low = text.lower()
        for fname, anchor, caption in CHARTS:
            if anchor in low and fname not in self.charts_done:
                path = CHART_DIR / fname
                if path.exists():
                    self.charts_done.add(fname)
                    return (path, caption)
        return None

    def embed_chart(self, path, caption):
        if self.get_y() > self.h - 120:
            self.add_page()
        self.ln(3)
        self.set_font("Body", "I", 8)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4, f"Figure: {caption}", align="C")
        self.ln(2)
        with Image.open(path) as img:
            iw, ih = img.size
        max_w = self.w - self.l_margin - self.r_margin - 20
        scale = min(max_w / iw, 120 / (ih * 0.264583))
        w_mm = iw * scale * 0.264583
        x = self.l_margin + (self.w - self.l_margin - self.r_margin - w_mm) / 2
        self.image(str(path), x=x, w=w_mm)
        self.ln(4)

    def place_display_math(self, expr):
        img, w_mm, h_mm = self.mr.render(expr, fontsize=11, display=True)
        max_w = self.w - self.l_margin - self.r_margin - 40
        if w_mm > max_w:
            scale = max_w / w_mm
            w_mm, h_mm = max_w, h_mm * scale
        if self.get_y() + h_mm + 5 > self.h - 20:
            self.add_page()
        self.ln(3)
        x = self.l_margin + (self.w - self.l_margin - self.r_margin - w_mm) / 2
        tmp = CACHE_DIR / f"_d{abs(hash(expr))}.png"
        img.save(tmp)
        self.image(str(tmp), x=x, w=w_mm, h=h_mm)
        self.ln(3)

    def place_inline_math(self, expr):
        img, w_mm, h_mm = self.mr.render(expr, fontsize=10.5, display=False)
        line_w = self.w - self.r_margin - self.get_x()
        if w_mm > line_w:
            self.ln(h_mm + 1)
            self.set_x(self.l_margin)
            line_w = self.w - self.l_margin - self.r_margin
        if w_mm > line_w:
            self.place_display_math(expr)
            return
        if self.get_y() + h_mm > self.h - 15:
            self.add_page()
        tmp = CACHE_DIR / f"_i{abs(hash(expr))}.png"
        img.save(tmp)
        self.image(str(tmp), x=self.get_x(), y=self.get_y() - h_mm * 0.12, w=w_mm, h=h_mm)
        self.set_x(self.get_x() + w_mm + 1.5)


# ═══════════════════════════════════════════════════════════════
# Markdown parser
# ═══════════════════════════════════════════════════════════════

def parse_blocks(text):
    blocks = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "":
            blocks.append(("blank", "")); i += 1; continue
        if line.strip().startswith("```"):
            code = []; i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(("code", "\n".join(code))); continue
        if re.match(r'^[-*_]{3,}\s*$', line.strip()):
            blocks.append(("hr", "")); i += 1; continue
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            blocks.append(("heading", (len(m.group(1)), m.group(2)))); i += 1; continue
        if line.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                r = lines[i].strip()
                if not re.match(r'^\|[\s\-:|]+\|$', r):
                    rows.append(r)
                i += 1
            if rows: blocks.append(("table", rows))
            continue
        if line.startswith(">"):
            q = []
            while i < len(lines) and lines[i].startswith(">"):
                q.append(lines[i].lstrip("> "))
                i += 1
            blocks.append(("blockquote", "\n".join(q))); continue
        lm = re.match(r'^(\s*)[-*+]\s+(.+)', line) or re.match(r'^(\s*)\d+\.\s+(.+)', line)
        if lm:
            items = []; pl = len(lm.group(1))
            while i < len(lines):
                m2 = re.match(rf'^(\s{{{pl}}})[-*+]\s+(.+)', lines[i]) or \
                     re.match(rf'^(\s{{{pl}}})\d+\.\s+(.+)', lines[i])
                if m2: items.append(m2.group(2)); i += 1
                elif lines[i].strip() == "":
                    i += 1
                    if i < len(lines):
                        m3 = re.match(rf'^(\s{{{pl}}})[-*+]\s+(.+)', lines[i]) or \
                             re.match(rf'^(\s{{{pl}}})\d+\.\s+(.+)', lines[i])
                        if m3: items.append(m3.group(2)); i += 1; continue
                    break
                else: break
            blocks.append(("list", items)); continue
        para = []
        while i < len(lines) and lines[i].strip() != "" and \
              not lines[i].startswith("#") and not lines[i].startswith(">") and \
              not lines[i].startswith("```") and \
              not re.match(r'^[-*+]\s', lines[i]) and \
              not re.match(r'^\d+\.\s', lines[i]) and \
              not lines[i].strip().startswith("|"):
            para.append(lines[i]); i += 1
        if para: blocks.append(("paragraph", " ".join(para)))
        continue
    return blocks


def split_segments(text):
    """Split into (text, bold, italic, is_math) segments."""
    segs = []
    for part in re.split(r'(\$[^$]+\$)', text):
        if part.startswith("$") and part.endswith("$"):
            segs.append((part[1:-1], False, False, True))
        else:
            for sp in re.split(r'(\*\*.*?\*\*|\*[^*]+\*)', part):
                if sp.startswith("**") and sp.endswith("**"):
                    segs.append((sp[2:-2], True, False, False))
                elif sp.startswith("*") and sp.endswith("*"):
                    segs.append((sp[1:-1], False, True, False))
                else:
                    segs.append((sp, False, False, False))
    return segs


# ═══════════════════════════════════════════════════════════════
# Renderers
# ═══════════════════════════════════════════════════════════════

def write_segmented(pdf, text, indent=0, size=10, lead=None):
    if lead is None: lead = size * 0.55
    line_w = pdf.w - pdf.l_margin - pdf.r_margin - indent
    for seg_text, bold, italic, is_math in split_segments(text):
        if is_math:
            pdf.place_inline_math(seg_text)
        else:
            style = "B" * bold + "I" * italic or ""
            pdf.set_font("Body", style, size)
            pdf.set_text_color(0)
            for j, sub in enumerate(seg_text.split("\n")):
                if j > 0: pdf.ln(lead)
                pdf.set_x(pdf.l_margin + indent)
                pdf.multi_cell(line_w, lead, sub)


def render_paragraph(pdf, text, indent=0):
    if "$$" not in text:
        pdf.ln(1)
        write_segmented(pdf, text, indent)
        pdf.ln(2)
        return
    parts = text.split("$$")
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            if part.strip():
                pdf.ln(1)
                write_segmented(pdf, part, indent)
                pdf.ln(1)
        else:
            expr = part.strip()
            if expr:
                pdf.place_display_math(expr)


def render_blockquote(pdf, text):
    if "$$" in text:
        for idx, part in enumerate(text.split("$$")):
            if idx % 2 == 0:
                if part.strip(): write_segmented(pdf, part, indent=8, size=9, lead=5)
            else:
                if part.strip(): pdf.place_display_math(part.strip())
    else:
        write_segmented(pdf, text, indent=8, size=9, lead=5)


# ═══════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════

def build_pdf():
    mr = MathRenderer(dpi=200)
    text = INPUT_FILE.read_text(encoding="utf-8")
    blocks = parse_blocks(text)
    pdf = ArticlePDF(mr)
    pdf.add_page()

    # Title
    pdf.ln(30)
    pdf.set_font("Body", "B", 22)
    pdf.set_text_color(20, 20, 80)
    pdf.multi_cell(0, 10, "Black Holes, Gravitational Waves,\nand the Hypothesis of\nSuperluminal Observers", align="C")
    pdf.ln(10)
    pdf.set_font("Body", "I", 12)
    pdf.set_text_color(80)
    pdf.cell(0, 8, "A Research Roadmap with Computational Analysis", align="C")
    pdf.ln(15)
    pdf.set_font("Body", "", 9)
    pdf.set_text_color(100)
    pdf.cell(0, 6, "Generated: June 2026", align="C")
    pdf.ln(8)
    pdf.cell(0, 6, "Math rendered via matplotlib mathtext  |  Charts from companion scripts", align="C")
    pdf.ln(18)
    pdf.set_font("Body", "B", 10)
    pdf.set_text_color(40)
    pdf.cell(0, 6, "Contents")
    pdf.ln(8)
    pdf.set_font("Body", "", 10)
    for item in [
        "I.   Established Constraints",
        "II.  Dragan-Ekert Framework",
        "III. Three Exploratory Hypotheses",
        "     III.1  Hypothesis I: Black Hole Cores",
        "     III.2  Hypothesis II: Dark Matter",
        "     III.3  Hypothesis III: Baryogenesis",
        "     III.4-III.12  Supporting Calculations",
        "IV.  Challenges and Open Problems",
        "V.   Conclusion",
    ]:
        pdf.cell(0, 5.5, item); pdf.ln(5.5)

    # Body
    for btype, content in blocks:
        if btype == "blank":
            pdf.ln(2)
        elif btype == "hr":
            pdf.ln(2)
            y = pdf.get_y()
            pdf.set_draw_color(180)
            pdf.line(pdf.l_margin + 30, y, pdf.w - pdf.r_margin - 30, y)
            pdf.ln(3)
        elif btype == "heading":
            level, title = content
            if level == 1: pdf.add_page()
            sizes = {1: 16, 2: 13, 3: 11, 4: 10, 5: 9, 6: 9}
            pdf.ln(4)
            pdf.set_font("Body", "B", sizes.get(level, 10))
            c = (20, 20, 80) if level <= 2 else (40, 40, 40)
            pdf.set_text_color(*c)
            clean = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', title)
            pdf.multi_cell(0, sizes.get(level, 10) * 0.7, clean)
            pdf.ln(2)
            pdf.set_text_color(0)
            ch = pdf.try_chart(title)
            if ch: pdf.embed_chart(*ch)
        elif btype == "paragraph":
            ch = pdf.try_chart(content)
            if ch: pdf.embed_chart(*ch)
            render_paragraph(pdf, content)
        elif btype == "blockquote":
            ch = pdf.try_chart(content)
            if ch: pdf.embed_chart(*ch)
            pdf.ln(2)
            y0 = pdf.get_y()
            pdf.set_font("Body", "I", 9)
            pdf.set_text_color(60, 60, 80)
            pdf.set_x(pdf.l_margin + 8)
            render_blockquote(pdf, content)
            y1 = pdf.get_y()
            pdf.set_draw_color(80, 80, 160)
            pdf.set_line_width(0.8)
            pdf.line(pdf.l_margin + 3, y0, pdf.l_margin + 3, y1 + 2)
            pdf.set_line_width(0.2)
            pdf.ln(3)
            pdf.set_text_color(0)
        elif btype == "code":
            pdf.ln(2)
            pdf.set_font("Mono", "", 7)
            pdf.set_text_color(60, 60, 60)
            for cl in content.split("\n"):
                pdf.set_x(pdf.l_margin + 10)
                pdf.cell(pdf.w - pdf.l_margin - pdf.r_margin - 20, 4, cl[:120],
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_text_color(0)
        elif btype == "list":
            pdf.ln(1)
            for item in content:
                pdf.set_x(pdf.l_margin + 8)
                pdf.set_font("Body", "", 10)
                pdf.cell(5, 5.5, chr(8226))
                pdf.set_x(pdf.l_margin + 14)
                write_segmented(pdf, item, indent=14, size=10, lead=5)
                pdf.ln(1)
            pdf.ln(2)
        elif btype == "table":
            pdf.ln(2)
            rows = [[c.strip() for c in r.strip("|").split("|")] for r in content]
            if rows:
                ncols = max(len(r) for r in rows)
                col_w = (pdf.w - pdf.l_margin - pdf.r_margin - 10) / ncols
                pdf.set_font("Body", "", 7)
                if len(rows) > 1:
                    pdf.set_fill_color(40, 40, 100)
                    pdf.set_text_color(255)
                    for cell in rows[0]:
                        pdf.cell(col_w, 6, cell[:35], border=1, fill=True)
                    pdf.ln()
                    pdf.set_text_color(0)
                    for row in rows[1:]:
                        for j, cell in enumerate(row):
                            if j >= ncols: break
                            fc = (248, 248, 252) if j % 2 == 0 else (255, 255, 255)
                            pdf.set_fill_color(*fc)
                            pdf.cell(col_w, 5, cell[:60], border=1, fill=True)
                        pdf.ln()
            pdf.ln(3)

    out = OUTPUT_FILE
    try:
        pdf.output(str(out))
    except PermissionError:
        out = OUTPUT_FILE.with_name("deepArticle_v2.pdf")
        pdf.output(str(out))
        print(f"NOTE: deepArticle.pdf is locked (close your PDF viewer).")
        print(f"Written to: {out}")
    size_kb = out.stat().st_size / 1024
    print(f"PDF written: {out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build_pdf()
