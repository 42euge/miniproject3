#!/usr/bin/env python3
"""Convert slides.md to a Beamer .tex file."""

import re
import sys

PREAMBLE = r"""\documentclass[aspectratio=169]{beamer}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{tikz}
\usetikzlibrary{positioning, arrows.meta}

\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{}
\setbeamertemplate{headline}{}
\setbeamercovered{transparent}

\definecolor{bg}{HTML}{000000}
\definecolor{text}{HTML}{E7F0FF}
\definecolor{muted}{HTML}{8EA3BE}
\definecolor{accent}{HTML}{5FD7FF}
\definecolor{accent2}{HTML}{9DFFCE}
\definecolor{warm}{HTML}{FFB86B}
\definecolor{bad}{HTML}{FF7A7A}
\definecolor{good}{HTML}{4CD964}

\setbeamercolor{normal text}{fg=text,bg=bg}
\setbeamercolor{frametitle}{fg=accent,bg=bg}
\setbeamercolor{background canvas}{bg=bg}
\setbeamercolor{structure}{fg=accent}
\setbeamercolor{itemize item}{fg=accent2}
\setbeamersize{text margin left=10mm, text margin right=10mm}
\setbeamertemplate{background canvas}{\color{bg}\rule{\paperwidth}{\paperheight}}
\setbeamertemplate{frametitle}{
    \vspace{0.08cm}
    {\usebeamerfont{frametitle}\color{accent}\insertframetitle}
    \par\nointerlineskip\color{muted}\hrule height 0.5pt\par
    \vspace{0.12cm}
}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}

\newcommand{\flowpoint}[1]{\vspace{0.12cm}\par\hangindent=0.5cm\hangafter=1\textbullet\ \textcolor{text}{#1}}

\begin{document}
"""

POSTAMBLE = r"\end{document}" + "\n"


def inline_transforms(line):
    """Apply inline Markdown → LaTeX transforms."""
    # $$...$$ display math on a single line → \[...\]
    line = re.sub(r'^\s*\$\$(.+?)\$\$\s*$', r'\\[\\1\\]', line)
    # Bold **text**
    line = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', line)
    # Italic *text* (not **)
    line = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\textit{\1}', line)
    # Inline code `text`
    line = re.sub(r'`(.+?)`', r'\\texttt{\1}', line)
    # Inline math $...$ (not $$)
    line = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', r'\\(\1\\)', line)
    return line


def is_display_math(line):
    return re.match(r'^\s*\$\$.+\$\$\s*$', line) or line.strip() in ('$$',)


def strip_yaml(text):
    """Remove YAML front matter if present."""
    if text.startswith('---'):
        end = text.find('\n---', 3)
        if end != -1:
            return text[end + 4:].lstrip('\n')
    return text


def split_slides(text):
    """Split on bare '---' lines (not inside fences)."""
    slides = []
    current = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith('```'):
            in_fence = not in_fence
        if not in_fence and re.match(r'^---\s*$', line):
            slides.append('\n'.join(current))
            current = []
        else:
            current.append(line)
    if current:
        slides.append('\n'.join(current))
    return [s.strip() for s in slides if s.strip()]


def chunk_body(body_lines):
    """Split body into blank-line-separated chunks."""
    chunks = []
    current = []
    for line in body_lines:
        if line == '':
            if current:
                chunks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return chunks


def render_chunk(lines, step, counters):
    """
    Render a single chunk (paragraph, list, math block, tikz, comment block).
    Returns (latex_string, new_step).
    """
    text = '\n'.join(lines)

    # HTML comment directives — handled at a higher level; skip lone comment lines
    if re.match(r'^\s*<!--.*-->\s*$', text, re.DOTALL):
        return ('', step)

    # Tikz fence passthrough
    if lines[0].startswith('```tikz') or lines[0] == '```tikz':
        inner = '\n'.join(lines[1:])
        inner = inner.rstrip('`').rstrip()
        return (inner + '\n', step)

    # Generic code fence passthrough
    if lines[0].startswith('```'):
        inner = '\n'.join(lines[1:])
        inner = re.sub(r'```\s*$', '', inner).rstrip()
        return (inner + '\n', step)

    # Display math block (standalone $$ ... $$)
    if len(lines) == 1 and re.match(r'^\s*\$\$(.+)\$\$\s*$', lines[0]):
        math = re.match(r'^\s*\$\$(.+)\$\$\s*$', lines[0]).group(1)
        out = f'    \\[\n        \\uncover<{step}->{{{math}}}\n    \\]\n'
        return (out, step + 1)

    # Multi-line display math $$ ... $$
    if lines[0].strip() == '$$' and lines[-1].strip() == '$$':
        inner = '\n'.join(lines[1:-1])
        out = f'    \\[\n        \\uncover<{step}->{{{inner}}}\n    \\]\n'
        return (out, step + 1)

    # Bullet list
    if lines[0].startswith('- ') or lines[0].startswith('* '):
        out = '    \\begin{itemize}\n'
        for line in lines:
            m = re.match(r'^[-*]\s+(.*)', line)
            if m:
                content = inline_transforms(m.group(1))
                out += f'        \\item<{step}-> {content}\n'
                step += 1
        out += '    \\end{itemize}\n'
        return (out, step)

    # Numbered list
    if re.match(r'^\d+\.\s', lines[0]):
        out = '    \\begin{enumerate}\n'
        for line in lines:
            m = re.match(r'^\d+\.\s+(.*)', line)
            if m:
                content = inline_transforms(m.group(1))
                out += f'        \\item<{step}-> {content}\n'
                step += 1
        out += '    \\end{enumerate}\n'
        return (out, step)

    # Raw LaTeX lines (start with \)
    if all(l.strip().startswith('\\') or l.strip() == '' for l in lines):
        out = ''
        for line in lines:
            if line.strip():
                out += f'    \\uncover<{step}->{{{line.strip()}}}\n'
                step += 1
        return (out, step)

    # Plain paragraph — wrap in \uncover
    transformed = ' '.join(inline_transforms(l) for l in lines if l.strip())
    out = f'    \\uncover<{step}->{{{transformed}}}\n'
    return (out, step + 1)


def render_columns_block(block_lines, step):
    """Render a <!-- columns --> ... <!-- /columns --> block."""
    # Split into cols
    cols = []
    current = []
    for line in block_lines:
        if re.match(r'^\s*<!--\s*col\s*-->\s*$', line):
            if current:
                cols.append(current)
            current = []
        else:
            current.append(line)
    if current:
        cols.append(current)

    out = '    \\begin{columns}[T]\n'
    for col_lines in cols:
        out += '        \\begin{column}{0.48\\textwidth}\n'
        chunks = chunk_body(col_lines)
        for chunk in chunks:
            rendered, step = render_chunk(chunk, step, {})
            # indent
            for ln in rendered.splitlines():
                out += '    ' + ln + '\n'
        out += '        \\end{column}\n'
    out += '    \\end{columns}\n'
    return out, step


def render_center_block(block_lines, step):
    """Render a <!-- center --> ... <!-- /center --> block."""
    out = '    \\begin{center}\n'
    chunks = chunk_body(block_lines)
    for index, chunk in enumerate(chunks):
        rendered, step = render_chunk(chunk, step, {})
        for ln in rendered.splitlines():
            out += '    ' + ln + '\n'
        if index < len(chunks) - 1:
            out += '    \\par\n'
    out += '    \\end{center}\n'
    return out, step


def render_slide(slide_text):
    """Convert a single slide block to LaTeX."""
    lines = slide_text.splitlines()

    # Determine frame title
    title = None
    body_start = 0
    if lines and re.match(r'^##\s+', lines[0]):
        title = re.match(r'^##\s+(.*)', lines[0]).group(1).strip()
        body_start = 1

    if title:
        out = f'\\begin{{frame}}{{{title}}}\n'
    else:
        out = '\\begin{frame}[plain]\n    \\vfill\n'

    body_lines = lines[body_start:]
    # Strip leading blank lines
    while body_lines and body_lines[0].strip() == '':
        body_lines.pop(0)

    step = 1

    # Process body with awareness of block-level comment directives
    i = 0
    while i < len(body_lines):
        line = body_lines[i]

        # <!-- columns --> block
        if re.match(r'^\s*<!--\s*columns\s*-->\s*$', line):
            block = []
            i += 1
            while i < len(body_lines) and not re.match(r'^\s*<!--\s*/columns\s*-->\s*$', body_lines[i]):
                block.append(body_lines[i])
                i += 1
            rendered, step = render_columns_block(block, step)
            out += rendered
            i += 1
            continue

        # <!-- center --> block
        if re.match(r'^\s*<!--\s*center\s*-->\s*$', line):
            block = []
            i += 1
            while i < len(body_lines) and not re.match(r'^\s*<!--\s*/center\s*-->\s*$', body_lines[i]):
                block.append(body_lines[i])
                i += 1
            rendered, step = render_center_block(block, step)
            out += rendered
            i += 1
            continue

        # tikz fence block — collect until closing ```
        if line.startswith('```tikz') or (line.strip() == '```tikz'):
            fence_lines = [line]
            i += 1
            while i < len(body_lines):
                fence_lines.append(body_lines[i])
                if body_lines[i].strip() == '```':
                    i += 1
                    break
                i += 1
            rendered, step = render_chunk(fence_lines, step, {})
            out += rendered
            continue

        # Generic fence block
        if line.startswith('```'):
            fence_lines = [line]
            i += 1
            while i < len(body_lines):
                fence_lines.append(body_lines[i])
                if body_lines[i].strip() == '```':
                    i += 1
                    break
                i += 1
            rendered, step = render_chunk(fence_lines, step, {})
            out += rendered
            continue

        # Blank line — advance
        if line.strip() == '':
            i += 1
            continue

        # Collect a paragraph/list chunk (until blank line or directive)
        chunk = []
        while i < len(body_lines):
            l = body_lines[i]
            if l.strip() == '':
                break
            if re.match(r'^\s*<!--', l):
                break
            if l.startswith('```'):
                break
            chunk.append(l)
            i += 1

        if chunk:
            rendered, step = render_chunk(chunk, step, {})
            out += rendered

    if not title:
        out += '    \\vfill\n'
    out += '\\end{frame}\n'
    return out


def convert(src_path, dst_path):
    with open(src_path, 'r') as f:
        text = f.read()

    text = strip_yaml(text)
    slides = split_slides(text)

    with open(dst_path, 'w') as f:
        f.write(PREAMBLE)
        f.write('\n')
        for slide in slides:
            f.write(render_slide(slide))
            f.write('\n')
        f.write(POSTAMBLE)

    print(f"Written {len(slides)} slides to {dst_path}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.md> <output.tex>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
