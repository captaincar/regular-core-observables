import re

with open(r'd:\DEV\fizyka\manuscript.md', 'r', encoding='utf-8') as f:
    md = f.read()

lines = md.split('\n')
output = []
i = 0

def escape_html(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

def process_inline(s):
    """Process inline content: save math, escape HTML, apply formatting, restore math."""
    math_spans = []
    def save_math(m):
        math_spans.append(m.group(0))
        return f'\x00MATH{len(math_spans)-1}\x00'

    s = re.sub(r'\$[^$]+\$', save_math, s)
    s = escape_html(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'\*(.+?)\*', r'<em>\1</em>', s)
    for idx, m in enumerate(math_spans):
        s = s.replace(f'\x00MATH{idx}\x00', m)
    return s

while i < len(lines):
    line = lines[i]
    stripped = line.strip()

    if not stripped:
        output.append('')
        i += 1
        continue

    # Display math - single-line $$...$$
    if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
        math = stripped[2:-2].strip()
        output.append(f'<div class="equation">\\[{math}\\]</div>')
        i += 1
        continue

    # Display math - multiline $$...$$
    if stripped == '$$':
        math_lines = []
        i += 1
        while i < len(lines) and lines[i].strip() != '$$':
            math_lines.append(lines[i].strip())
            i += 1
        i += 1
        math = ' '.join(math_lines)
        output.append(f'<div class="equation">\\[{math}\\]</div>')
        continue

    # Horizontal rule
    if stripped == '---':
        output.append('<hr>')
        i += 1
        continue

    # Headings
    heading_match = None
    for level, tag in [(4, 'h4'), (3, 'h3'), (2, 'h2'), (1, 'h1')]:
        m = re.match(r'^' + '#' * level + r' (.+)$', stripped)
        if m:
            heading_match = (tag, m.group(1))
            break

    if heading_match:
        tag, text = heading_match
        output.append(f'<{tag}>{process_inline(text)}</{tag}>')
        i += 1
        continue

    # Tables
    if stripped.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i+1].strip()):
        def split_table_row(row):
            """Split table row on | but not inside $...$."""
            math_saved = []
            def save_math_tbl(m):
                math_saved.append(m.group(0))
                return f'\x00MTBL{len(math_saved)-1}\x00'
            row = re.sub(r'\$[^$]+\$', save_math_tbl, row)
            cells = [c.strip() for c in row.split('|')[1:-1]]
            # Restore math
            restored = []
            for cell in cells:
                for idx, m in enumerate(math_saved):
                    cell = cell.replace(f'\x00MTBL{idx}\x00', m)
                restored.append(cell)
            return restored

        cells = [process_inline(c) for c in split_table_row(stripped)]
        header_html = '<thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead>'
        i += 2
        body_rows = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            cells = [process_inline(c) for c in split_table_row(lines[i].strip())]
            body_rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
            i += 1
        output.append('<table>' + header_html + '<tbody>' + ''.join(body_rows) + '</tbody></table>')
        continue

    # Ordered lists
    m = re.match(r'^(\d+)\.\s+(.+)$', stripped)
    if m:
        items = []
        while i < len(lines):
            m2 = re.match(r'^(\d+)\.\s+(.+)$', lines[i].strip())
            if not m2:
                break
            items.append(process_inline(m2.group(2)))
            i += 1
        output.append('<ol>' + ''.join(f'<li>{item}</li>' for item in items) + '</ol>')
        continue

    # Unordered lists
    m = re.match(r'^-\s+(.+)$', stripped)
    if m:
        items = []
        while i < len(lines):
            m2 = re.match(r'^-\s+(.+)$', lines[i].strip())
            if not m2:
                break
            items.append(process_inline(m2.group(1)))
            i += 1
        output.append('<ul>' + ''.join(f'<li>{item}</li>' for item in items) + '</ul>')
        continue

    # Image
    m = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', stripped)
    if m:
        alt, src = m.group(1), m.group(2)
        output.append(f'<figure><img src="{src}" alt="{escape_html(alt)}"><figcaption>{escape_html(alt)}</figcaption></figure>')
        i += 1
        continue

    # Regular paragraph
    output.append(f'<p>{process_inline(stripped)}</p>')
    i += 1

html_body = '\n'.join(output)
html_body = re.sub(r'<p>\s*</p>', '', html_body)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Regular Black Hole and Compact Star Phenomenology with a Negative-Pressure Core</title>
<script>
MathJax = {{
  tex: {{
    inlineMath: [['$', '$']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true
  }}
}};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
body {{
  max-width: 820px;
  margin: 40px auto;
  padding: 20px 40px;
  font-family: 'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif;
  font-size: 17px;
  line-height: 1.65;
  color: #222;
  background: #fdfdfd;
}}
h1 {{
  font-size: 1.8em;
  text-align: center;
  margin-bottom: 0.2em;
  line-height: 1.3;
}}
h2 {{
  font-size: 1.35em;
  margin-top: 2em;
  border-bottom: 1px solid #ccc;
  padding-bottom: 0.2em;
}}
h3 {{
  font-size: 1.15em;
  margin-top: 1.6em;
}}
h4 {{
  font-size: 1.05em;
  margin-top: 1.3em;
}}
.equation {{
  margin: 1.2em 0;
  text-align: center;
  overflow-x: auto;
}}
table {{
  border-collapse: collapse;
  margin: 1em auto;
  font-size: 0.92em;
}}
th, td {{
  padding: 6px 14px;
  border: 1px solid #ccc;
  text-align: left;
}}
th {{
  background: #f0f0f0;
  font-weight: 600;
}}
thead th {{
  border-bottom: 2px solid #999;
}}
code {{
  background: #f0f0f0;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 0.9em;
}}
hr {{
  border: none;
  border-top: 1px solid #ddd;
  margin: 1.5em 0;
}}
figure {{
  text-align: center;
  margin: 1.5em 0;
}}
figcaption {{
  font-size: 0.88em;
  color: #666;
  margin-top: 0.3em;
}}
img {{
  max-width: 100%;
}}
ol, ul {{
  padding-left: 1.8em;
}}
@media (max-width: 600px) {{
  body {{ padding: 10px 15px; font-size: 15px; }}
  table {{ font-size: 0.8em; }}
  th, td {{ padding: 4px 6px; }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>'''

with open(r'd:\DEV\fizyka\manuscript.html', 'w', encoding='utf-8') as f:
    f.write(html)

print(f'HTML written to manuscript.html ({len(html)} bytes)')
