import re


def escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def process_inline(text):
    """Process inline content: save math, escape HTML, apply formatting, restore math."""
    math_spans = []

    def save_math(match):
        math_spans.append(match.group(0))
        return f'\x00MATH{len(math_spans)-1}\x00'

    text = re.sub(r'\$[^$]+\$', save_math, text)
    text = escape_html(text)
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    for idx, math_span in enumerate(math_spans):
        text = text.replace(f'\x00MATH{idx}\x00', math_span)
    return text


def split_table_row(row):
    """Split table row on | but not inside $...$."""
    math_saved = []

    def save_math_tbl(match):
        math_saved.append(match.group(0))
        return f'\x00MTBL{len(math_saved)-1}\x00'

    row = re.sub(r'\$[^$]+\$', save_math_tbl, row)
    cells = [cell.strip() for cell in row.split('|')[1:-1]]
    restored = []
    for cell in cells:
        for idx, math_span in enumerate(math_saved):
            cell = cell.replace(f'\x00MTBL{idx}\x00', math_span)
        restored.append(cell)
    return restored


def render_markdown_to_html(md_text, title, table_font_size):
    lines = md_text.split('\n')
    output = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped:
            output.append('')
            index += 1
            continue

        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            math = stripped[2:-2].strip()
            output.append(f'<div class="equation">\\[{math}\\]</div>')
            index += 1
            continue

        if stripped == '$$':
            math_lines = []
            index += 1
            while index < len(lines) and lines[index].strip() != '$$':
                math_lines.append(lines[index].strip())
                index += 1
            index += 1
            math = ' '.join(math_lines)
            output.append(f'<div class="equation">\\[{math}\\]</div>')
            continue

        if stripped == '---':
            output.append('<hr>')
            index += 1
            continue

        heading_match = None
        for level, tag in [(4, 'h4'), (3, 'h3'), (2, 'h2'), (1, 'h1')]:
            match = re.match(r'^' + '#' * level + r' (.+)$', stripped)
            if match:
                heading_match = (tag, match.group(1))
                break

        if heading_match:
            tag, text = heading_match
            output.append(f'<{tag}>{process_inline(text)}</{tag}>')
            index += 1
            continue

        if stripped.startswith('|') and index + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[index + 1].strip()):
            cells = [process_inline(cell) for cell in split_table_row(stripped)]
            header_html = '<thead><tr>' + ''.join(f'<th>{cell}</th>' for cell in cells) + '</tr></thead>'
            index += 2
            body_rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [process_inline(cell) for cell in split_table_row(lines[index].strip())]
                body_rows.append('<tr>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
                index += 1
            output.append('<table>' + header_html + '<tbody>' + ''.join(body_rows) + '</tbody></table>')
            continue

        match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if match:
            items = []
            while index < len(lines):
                item_match = re.match(r'^(\d+)\.\s+(.+)$', lines[index].strip())
                if not item_match:
                    break
                items.append(process_inline(item_match.group(2)))
                index += 1
            output.append('<ol>' + ''.join(f'<li>{item}</li>' for item in items) + '</ol>')
            continue

        match = re.match(r'^-\s+(.+)$', stripped)
        if match:
            items = []
            while index < len(lines):
                item_match = re.match(r'^-\s+(.+)$', lines[index].strip())
                if not item_match:
                    break
                items.append(process_inline(item_match.group(1)))
                index += 1
            output.append('<ul>' + ''.join(f'<li>{item}</li>' for item in items) + '</ul>')
            continue

        match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', stripped)
        if match:
            alt_text, src = match.group(1), match.group(2)
            output.append(
                f'<figure><img src="{src}" alt="{escape_html(alt_text)}"><figcaption>{escape_html(alt_text)}</figcaption></figure>'
            )
            index += 1
            continue

        output.append(f'<p>{process_inline(stripped)}</p>')
        index += 1

    html_body = '\n'.join(output)
    html_body = re.sub(r'<p>\s*</p>', '', html_body)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
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
  font-size: {table_font_size};
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


def build_html(md_path, html_path, title, table_font_size):
    with open(md_path, 'r', encoding='utf-8') as markdown_file:
        markdown = markdown_file.read()

    html = render_markdown_to_html(markdown, title, table_font_size)

    with open(html_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html)

    print(f'HTML written to {html_path} ({len(html)} bytes)')