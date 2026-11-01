#!/usr/bin/env python3
"""Sinh dự án LaTeX (Overleaf-ready) từ các báo cáo markdown + artifact.

Bảng số liệu trong .tex đến từ .md -- vốn đã được `scripts/verify_replication.py`
đối chiếu máy với artifact.  Hình vẽ/sơ đồ (TikZ + pgfplots, không matplotlib) đọc
TRỰC TIẾP artifact và inline dữ liệu vào .tex, nên mỗi hình mang theo đường dẫn
artifact + seed của chính nó.

Dùng:
    .venv/bin/python python/make_latex_report.py            # sinh latex/
    .venv/bin/python python/make_latex_report.py --check    # thêm: biên dịch thử (bản ASCII)
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import math
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DEFAULT = ROOT / "latex"

REPORTS = [
    # Chuoi pre-registration: duoc giu nguyen thu tu de thay protocol khoa truoc khi chay.
    # 06_full_research_report.md co y LOAI: do la ban ket qua cu, da bi 09 thay the;
    # gop vao cung mot PDF se de hai so lieu mau thuan trong cung mot tai lieu.
    "research_report/00_research_plan.md",
    "research_report/01_literature_matrix.md",
    "research_report/02_research_gap_analysis.md",
    "research_report/03_research_questions.md",
    "research_report/04_methodology_rationale.md",
    "research_report/05_experimental_design.md",
    "research_report/07_open_issues.md",
    "research_report/08_advisor_meeting_brief.md",
    "research_report/09_final_research_report.md",
    "EXPERIMENT_RESULTS.md",
    "GROUP_IMPORTANCE_REPORT.md",
    "ABLATION_REPORT.md",
    "STABILITY_REPORT.md",
    "SAMPLER_DIAGNOSTICS.md",
    "REPRODUCIBILITY.md",
    "TECHNICAL_AUDIT.md",
]

CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
TOTAL_FEATURES = 55
GROUPS_ORDER = ["handles", "ldrmodules", "malfind", "dlllist", "svcscan",
                "pslist", "callbacks", "psxview", "modules"]

# --------------------------------------------------------------------------- #
# Ký hiệu
# --------------------------------------------------------------------------- #
SYMBOLS = [
    ("Δ", r"$\Delta$"), ("δ", r"$\delta$"), ("ρ", r"$\rho$"), ("µ", r"$\mu$"),
    ("μ", r"$\mu$"), ("σ", r"$\sigma$"), ("α", r"$\alpha$"), ("β", r"$\beta$"),
    ("λ", r"$\lambda$"), ("χ", r"$\chi$"), ("≈", r"$\approx$"), ("≥", r"$\geq$"),
    ("≤", r"$\leq$"), ("≠", r"$\neq$"), ("→", r"$\rightarrow$"),
    ("←", r"$\leftarrow$"), ("↔", r"$\leftrightarrow$"), ("×", r"$\times$"),
    ("∝", r"$\propto$"), ("∈", r"$\in$"), ("∉", r"$\notin$"), ("∩", r"$\cap$"),
    ("∪", r"$\cup$"), ("∑", r"$\sum$"), ("∅", r"$\varnothing$"), ("∞", r"$\infty$"),
    ("·", r"$\cdot$"), ("±", r"$\pm$"),
    ("—", "---"), ("–", "--"), ("‑", "-"), ("\u00a0", " "), ("\u202f", " "),
    ("\u2009", " "), ("‘", "'"), ("’", "'"), ("“", "``"), ("”", "''"),
    ("•", r"$\bullet$"), ("✓", r"\checkmark"), ("✗", r"{\textcolor{red!70!black}{$\times$}}"),
    ("⩾", r"$\geq$"), ("⩽", r"$\leq$"),
    # bo sung sau loi bien dich U+03C6 va ket qua scan toan bo ma hoa phi-ASCII
    ("−", r"$-$"), ("§", r"\S"), ("…", r"\ldots"),
    ("φ", r"$\varphi$"), ("Φ", r"$\Phi$"), ("Σ", r"$\Sigma$"), ("σ", r"$\sigma$"),
    ("Ω", r"$\Omega$"), ("ω", r"$\omega$"), ("≫", r"$\gg$"), ("≪", r"$\ll$"),
    ("✅", r"{\color{green!45!black}\checkmark}"),
    ("❌", r"{\textcolor{red!70!black}{$\times$}}"),
    ("⚠", r"\textbf{(!)}"), ("️", ""),
]

# Trong lstlisting khong the chen macro LaTeX, nen ky hoan toan trong code block
# phai duoc thay bang chu ASCII (van giu nghia, van doc duoc tren May va Overleaf).
CODE_ASCII = [
    ("→", "->"), ("←", "<-"), ("↔", "<->"), ("≫", ">>"), ("≪", "<<"),
    ("≥", ">="), ("≤", "<="), ("≠", "!="), ("≈", "~="), ("±", "+/-"),
    ("×", "x"), ("·", "."), ("−", "-"), ("∝", "prop"), ("∑", "sum"),
    ("∏", "prod"), ("√", "sqrt"), ("∞", "inf"), ("∅", "empty"),
    ("∈", " in "), ("∉", " notin "), ("⊂", " subset "), ("∩", "^"), ("∪", "v"),
    ("Δ", "Delta"), ("δ", "delta"), ("ρ", "rho"), ("φ", "phi"), ("Φ", "Phi"),
    ("Σ", "Sigma"), ("σ", "sigma"), ("μ", "mu"), ("µ", "mu"), ("α", "alpha"),
    ("β", "beta"), ("λ", "lambda"), ("π", "pi"), ("χ", "chi"), ("ω", "omega"),
    ("Ω", "Omega"), ("§", "S"), ("—", "--"), ("–", "-"), ("‑", "-"),
    ("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'), ("…", "..."),
    ("•", "*"), ("✓", "[OK]"), ("✅", "[OK]"), ("✗", "[X]"), ("❌", "[X]"),
    ("⚠", "(!)"), ("️", ""), (" ", " "), (" ", " "), (" ", " "),
    ("‑", "-"),
]


def ascii_math(s: str) -> str:
    for uni, rep in CODE_ASCII:
        if uni in s:
            s = s.replace(uni, rep)
    return s


CHAR_ESCAPES = [
    ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("<", r"\textless{}"), (">", r"\textgreater{}"), ("|", r"\textbar{}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
]

INLINE_TOKEN = re.compile(
    r"(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)]+\))"
)


def _escape_text(s: str) -> str:
    for raw, tex in CHAR_ESCAPES:
        s = s.replace(raw, tex)
    return s


def _symbols(s: str) -> str:
    for uni, tex in SYMBOLS:
        if uni in s:
            s = s.replace(uni, tex)
    return s


URL_RE = re.compile(r"https?://[^\s)\]}]+")
QUOTE_RE = re.compile(r'"([^"]{1,300}?)"')


def _render_text(s: str) -> str:
    s = QUOTE_RE.sub(r"``\1''", s)
    out, pos = [], 0
    for m in URL_RE.finditer(s):
        out.append(_symbols(_escape_text(s[pos:m.start()])))
        out.append(r"\nolinkurl{" + _escape_text(m.group(0)) + r"}")
        pos = m.end()
    out.append(_symbols(_escape_text(s[pos:])))
    return "".join(out)


BREAK_AFTER = [(r"\_", r"\_\allowbreak{}"), ("/", r"/\allowbreak{}"),
               (".", r".\allowbreak{}"), ("-", r"-\allowbreak{}")]


def _inline_code(s: str) -> str:
    inner = s[1:-1]
    # SHA-256 va cac chu hex/ID dai khong chua ky tu ngat: no vo thanh mot
    # hbox dai hon \textwidth, tao overfull 150-170pt. Cho pheut giat 16 ky tu.
    if len(inner) >= 24 and not any(c in inner for c in "-_/. "):
        inner = "".join(c + ("\x01" if (i + 1) % 16 == 0 and i + 1 < len(inner) else "")
                        for i, c in enumerate(inner))
    body = _symbols(_escape_text(inner))
    for raw, rep in BREAK_AFTER:
        body = body.replace(raw, rep)
    return r"\texttt{" + body.replace("\x01", r"\allowbreak{}") + "}"


def inline(s: str) -> str:
    out, pos = [], 0
    for m in INLINE_TOKEN.finditer(s):
        out.append(_render_text(s[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(_inline_code(tok))
        elif tok.startswith("**"):
            out.append(r"\textbf{" + inline(tok[2:-2]) + "}")
        elif tok.startswith("["):
            mm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            label, url = mm.group(1), mm.group(2)
            if url.startswith(("http://", "https://", "doi:")):
                out.append(r"\href{%s}{%s}" % (url.replace("#", r"\#"), inline(label)))
            else:
                out.append(inline(label) + r"\,{\footnotesize(\texttt{"
                           + _escape_text(url) + r"})}")
        else:
            out.append(r"\emph{" + inline(tok[1:-1]) + "}")
        pos = m.end()
    out.append(_render_text(s[pos:]))
    return "".join(out)


# --------------------------------------------------------------------------- #
# Bảng
# --------------------------------------------------------------------------- #
NUMERIC_CELL = re.compile(r"^[\s±+\-~<≥≤≈*^0-9,.%/eE—–]+$")


def _cells(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _display_len(s: str) -> int:
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*|\*|\\", "", s)
    mm = re.match(r"\[([^\]]+)\]\([^)]+\)", s)
    return len(mm.group(1)) if mm else len(s)


def _aligns(sep: list[str], ncol: int) -> list[str]:
    out = []
    for i in range(ncol):
        s = sep[i] if i < len(sep) else "---"
        if s.endswith(":"):
            out.append("r")
        elif s.startswith(":"):
            out.append("l")
        else:
            out.append("l")
    return out


def _emit_table(header: list[str], sep: list[str], rows: list[list[str]]) -> str:
    """Tabular/longtable tu can chinh.

    Loi co dinh o day: cot `p{}` phai chi duoc phan con lai SAU khi tru di chieu
    dat cua cac cot ngan, neu khong to bang tron > \textwidth (213 overfull).
    """
    ncol = len(header)
    rows = [r[:ncol] + [""] * (ncol - len(r)) for r in rows]
    widths = [max(_display_len(header[i]), max((_display_len(r[i]) for r in rows), default=0))
              for i in range(ncol)]
    widths = [min(w, 60) for w in widths]
    numeric = []
    for i in range(ncol):
        vals = [r[i] for r in rows if r[i]]
        numeric.append(bool(vals) and all(NUMERIC_CELL.match(v) for v in vals))

    CHAR = 0.0069  # do dai 1 ky tu tinh theo \textwidth (uoc luong conservatve)
    narrow = sum(widths[i] * CHAR + 0.012 for i in range(ncol) if widths[i] <= 30)
    wide = [i for i in range(ncol) if widths[i] > 30]
    colspec: list[str] = []
    fracs: dict[int, float] = {}
    if wide:
        room = max(0.28, 0.90 - narrow)
        text_w = sum(widths[i] for i in wide)
        for i in range(ncol):
            if i in wide:
                frac = max(0.075, room * widths[i] / text_w)
                fracs[i] = frac
                colspec.append(r">{\RaggedRight\arraybackslash}p{%.3f\textwidth}" % frac)
            else:
                colspec.append("r" if numeric[i] else "l")
    else:
        colspec = ["r" if numeric[i] else "l" for i in range(ncol)]

    font = r"\footnotesize" if (ncol >= 7 or len(rows) >= 18) else r"\small"
    head = " & ".join(r"\textbf{%s}" % inline(h) for h in header)
    body = [" & ".join(inline(c) for c in r) + r" \\" for r in rows]
    spec = "@{}" + "@{\\hspace{6pt}}".join(colspec) + "@{}"

    # tabular khong cat trang duoc: chieu cao = tong (so dong boc nhatt tren 1 dong)
    lines_est = sum(
        max([1] + [max(1, math.ceil(len(r[i]) * CHAR / fracs[i])) for i in fracs if r[i]])
        for r in rows)
    if len(rows) > 26 or lines_est > 38:
        return "\n".join([
            r"{\footnotesize",
            r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
            r"\begin{longtable}{%s}" % spec,
            r"\toprule " + head + r" \\ \midrule \endhead",
            r"\endfoot\bottomrule",
            *body,
            r"\end{longtable}}",
        ]) + "\n"

    tab = "\n".join([
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{%s}" % spec,
        r"\toprule", head + r" \\", r"\midrule", *body, r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
    ])
    return ("\n".join([r"\begin{center}" + font, tab, r"\end{center}"]) + "\n")


# --------------------------------------------------------------------------- #
# Markdown -> LaTeX
# --------------------------------------------------------------------------- #
def md_to_tex(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    para: list[str] = []
    i, n = 0, len(lines)

    def flush():
        if para:
            out.append(inline(" ".join(para).strip()) + "\n")
            para.clear()

    while i < n:
        line = lines[i]
        s = line.strip()

        if s.startswith("```"):
            flush()
            lang = s[3:].strip()
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(ascii_math(lines[i]))
                i += 1
            i += 1
            style = "shell" if lang in ("bash", "sh", "zsh", "console") else "base"  # noqa: F841
            out.append("\\begin{codeblock}")
            out.extend(code)
            out.append(r"\end{codeblock}")
            out.append("")
            continue

        if not s:
            flush()
            i += 1
            continue

        h = re.match(r"^(#{1,6})\s+(.*)$", s)
        if h:
            flush()
            depth = len(h.group(1))
            cmd = {1: "section", 2: "subsection", 3: "subsubsection",
                   4: "paragraph", 5: "subparagraph", 6: "subparagraph"}[depth]
            title = inline(h.group(2))
            out.append(r"\%s%s{%s}" % (cmd, "*" if depth >= 4 else "", title))
            out.append("")
            i += 1
            continue

        if s in ("---", "***", "___"):
            flush()
            out.append(r"\par\medskip\hrule\medskip")
            i += 1
            continue

        if s.startswith("|") and i + 1 < n and re.match(r"^\|?[\s:|-]+\|?$", lines[i + 1].strip()):
            flush()
            header = _cells(s)
            sep = _cells(lines[i + 1].strip())
            i += 2
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i].strip()))
                i += 1
            out.append(_emit_table(header, sep, rows))
            continue

        if s.startswith(">"):
            flush()
            q: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                q.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(r"\begin{quote}" + "\n" + inline(" ".join(x for x in q if x)) +
                       "\n" + r"\end{quote}" + "\n")
            continue

        if re.match(r"^(\s*)([-*+]|\d+[.)])\s+", line):
            flush()
            items = []
            while i < n:
                cm = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$", lines[i])
                if cm:
                    items.append((len(cm.group(1)) // 2, bool(re.match(r"\d", cm.group(2))),
                                  cm.group(3).strip()))
                    i += 1
                elif items and re.match(r"^\s{2,}\S", lines[i]) and lines[i].strip():
                    items[-1] = items[-1][:2] + (items[-1][2] + " " + lines[i].strip(),)
                    i += 1
                else:
                    break
            out.append(_emit_list(items, 0, 0))
            continue

        para.append(s)
        i += 1

    flush()
    return "\n".join(out)


def _emit_list(items: list[tuple[int, bool, str]], start: int, depth: int) -> str:
    res: list[str] = []
    k = start
    while k < len(items):
        d, ordered, text = items[k]
        if d < depth:
            break
        if d > depth:
            res.append(r"\item " + inline(text))
            k += 1
            block = _emit_list(items, k, depth + 1)
            if block:
                res.append(block)
                k += _count(items[k:], depth + 1)
            continue
        env = "enumerate" if ordered else "itemize"
        if res and res[0].startswith(r"\begin{") and res[0][7:-1] != env:
            pass
        j = k
        while j < len(items) and items[j][0] == depth:
            j += 1
        # một khối cùng cấp: chọn env theo item đầu
        res.append(r"\begin{%s}" % env)
        for t in items[k:j]:
            res.append(r"\item " + inline(t[2]))
        res.append(r"\end{%s}" % env)
        k = j
    return "\n".join(res) + "\n" if res else ""


def _count(rest: list[tuple[int, bool, str]], depth: int) -> int:
    c = 0
    for d, _o, _t in rest:
        if d >= depth:
            c += 1
        else:
            break
    return c


# --------------------------------------------------------------------------- #
# Hình: TikZ / pgfplots đọc trực tiếp artifact
# --------------------------------------------------------------------------- #
def _src(*parts: str) -> str:
    body = " | ".join(parts)
    body = body.replace("{", "").replace("}", "")
    body = body.replace("%", r"\percent").replace("#", "no")
    return r"\typeout{RAM-EVID figure source: %s}" % body


def _g(v: float) -> str:
    return ("%.5f" % v).rstrip("0").rstrip(".")


def read_summary(run: str, file: str) -> pd.DataFrame | None:
    p = RESULTS / "group_experiments" / run / file
    if not p.exists():
        return None
    return pd.read_csv(p)


def fill(template: str, **tokens) -> str:
    """Thay token @@NAME@@ -- tra%nh % operator vi LaTeX day dang %."""
    for key, val in tokens.items():
        template = template.replace("@@%s@@" % key.upper(), str(val))
    if "@@" in template:
        raise ValueError("token chưa được thay: " + template[template.index("@@"):][:60])
    return template


def fig_estimand_pipeline() -> str:
    return r"""
\begin{figure}[t]\centering
""" + _src("so do khai niem; chua dung so lieu") + r"""
\begin{tikzpicture}[
  box/.style={draw=black!55, rounded corners=2pt, align=center, inner sep=5pt,
              minimum width=3.4cm, minimum height=1.1cm, font=\scriptsize},
  q/.style={align=center, font=\scriptsize, text=black!72},
  arr/.style={-{Latex[length=2.4mm]}, thick, draw=black!60},
]
\node[font=\sffamily\bfseries\footnotesize] at (0,4.75)
  {Cùng một model XGBoost đã fit --- bốn câu hỏi khác nhau, bốn estimand khác nhau};
\node[box, fill=blue!8]    (U) at (-5.3,3.15) {\textbf{UTILITY}\\ bỏ nhóm + \emph{học lại}\\ \texttt{run\_group\_ablation}};
\node[box, fill=teal!8]    (R) at (-1.8,3.15) {\textbf{RELIANCE}\\ phá nhóm trên \emph{model cố định}\\ \texttt{run\_group\_importance}};
\node[box, fill=orange!8]  (A) at (1.8,3.15)  {\textbf{ATTRIBUTION}\\ chia công theo liên minh\\ group Shapley (512)};
\node[box, fill=red!7]     (C) at (5.3,3.15)  {\textbf{CONDITIONAL}\\ giá trị còn lại sau conditioning\\ CFI/ARF};
\node[q] (uq) at (-5.3,1.55) {"Bỏ nhóm rồi học lại\\ thì mất bao nhiêu?"\\[3pt] $\Rightarrow$ hữu ích cho task\\ \emph{trong protocol}};
\node[q] (rq) at (-1.8,1.55) {"Đang dựa vào nó\\ nhiều tới đâu khi bị phá?"\\[3pt] $\Rightarrow$ phụ thuộc\\ \emph{của model đã fit}};
\node[q] (aq) at (1.8,1.55)  {"Model \emph{quy công} cho ai?"\\[3pt] $\Rightarrow$ không có đối chứng\\ khớp rộng, không có $p$};
\node[q] (cq) at (5.3,1.55)  {"Còn lại bao nhiêu sau khi\\ đã điều kiện hoá?"\\[3pt] $\Rightarrow$ chỉ dùng được nếu\\ sampler đạt diagnostics};
\draw[arr] (U) -- (uq); \draw[arr] (R) -- (rq);
\draw[arr] (A) -- (aq); \draw[arr] (C) -- (cq);
\node[box, fill=yellow!12] (S) at (0,-0.45) {\textbf{STABILITY}\\ đổi seed split / seed model / model khác};
\draw[arr] (uq.south) |- (S.west);
\draw[arr] (rq.south) |- (S.north west);
\draw[arr] (aq.south) |- (S.north east);
\draw[arr] (cq.south) |- (S.east);
\node[draw=red!60!black, dashed, rounded corners=2pt, inner sep=5pt, align=left,
      text width=8.4cm, font=\scriptsize, text=red!70!black, anchor=north west] at (-7.1,-1.35)
  {\textbf{Cấm hoán đổi kết luận.}\ \ SHAP cao $\neq$ thông tin riêng $\neq$ nguyên nhân.\
   Không có thiết kế can thiệp hợp lệ $\Rightarrow$ không dùng từ ``causal''.\\
   Không gộp $p$ qua seed/model; không pool null của các nhóm khác độ rộng.};
\end{tikzpicture}
\caption{Bốn estimand và phép đo stability, kèm câu hỏi mà mỗi cái trả lời. Mọi bảng trong báo cáo này được dán nhãn theo một trong bốn ô này.}
\end{figure}
"""


def fig_reliance_splits(seeds: dict[str, str]) -> str:
    per, srcs = {}, []
    for label, run in seeds.items():
        df = read_summary(run, "permutation_summary.csv")
        if df is None:
            srcs.append("%s: CHUA CO ARTIFACT" % run)
            continue
        per[label] = dict(zip(df["group"], df["delta_log_loss_mean"]))
        srcs.append("%s = results/group_experiments/%s/permutation_summary.csv (200 doi chung khop rong)"
                    % (label, run))
    if not per:
        return prov_missing(srcs)
    glist = [g for g in GROUPS_ORDER if all(g in d for d in per.values())]
    colors = {"split42": "blue!55!black", "split43": "teal!60!black", "split44": "orange!70!black"}
    plots, legends = [], []
    for label in per:
        coords = "".join("(%d,%s)" % (i + 1, _g(per[label][g])) for i, g in enumerate(glist))
        plots.append(r"\addplot+[only marks, mark=*, mark size=2.4pt, %s] coordinates {%s};"
                     % (colors.get(label, "black"), coords))
        legends.append(label)
    return fill(r"""
\begin{figure}[t]\centering
@@SRC@@
\begin{tikzpicture}
\begin{axis}[
  width=\textwidth, height=7.4cm,
  ylabel={$\Delta$ log-loss khi phá nhóm (càng cao càng quan trọng)},
  xlabel={Nhóm (sắp theo hạng của split 42)},
  xmin=0.4, xmax=@@XMAX@@, ymin=0, ymax=@@YMAX@@, xtick={@@XTICK@@},
  xticklabels={@@XTLAB@@},
  grid=major, grid style={black!10},
  tick label style={font=\footnotesize},
  legend style={at={(0.98,0.97)}, anchor=north east, font=\scriptsize, draw=black!20,
                fill=white, fill opacity=0.85},
]
@@PLOTS@@
\legend{@@LEGEND@@}
\end{axis}
\end{tikzpicture}
\caption{Reliance (block permutation trên model đã fit) cho chín nhóm ở ba split seed. Ba marker chồng khít nhau ở hầu hết nhóm --- đó chính là kết quả hạng \emph{bất biến} ($\rho=1{,}000$). \texttt{handles} cao gấp nhiều lần các nhóm còn lại; \texttt{modules} áp sát~0.}
\end{figure}
""", src=_src(*(srcs or ["khong co artifact"]), ),
            xmax="%d.6" % len(glist),
            ymax="%.2f" % (1.12 * max(v for d in per.values() for v in d.values())),
            xtick=",".join(str(i + 1) for i in range(len(glist))),
            xtlab=",".join(r"\texttt{%s}" % g for g in glist),
            plots="\n".join(plots), legend=", ".join(legends))


def fig_rank_drift() -> str:
    util, reli, srcs = {}, {}, []
    for seed, run in [("42", "ablation_v4"), ("43", "ablation_xgb_s43"), ("44", "ablation_xgb_s44")]:
        df = read_summary(run, "summary.csv")
        if df is None:
            srcs.append("utility %s: CHUA CO ARTIFACT" % run)
            continue
        util[seed] = dict(zip(df["group"], df["rank_log_loss"]))
        srcs.append("utility seed %s = results/group_experiments/%s/summary.csv (cot rank_log_loss)" % (seed, run))
    for seed, run in [("42", "importance_v3"), ("43", "importance_v3_s43"), ("44", "importance_v3_s44")]:
        df = read_summary(run, "permutation_summary.csv")
        if df is None:
            srcs.append("reliance %s: CHUA CO ARTIFACT" % run)
            continue
        reli[seed] = dict(zip(df["group"], df["rank"]))
        srcs.append("reliance seed %s = results/group_experiments/%s/permutation_summary.csv (cot rank)" % (seed, run))
    srcs.append("huong cot: rank 1 = nhom quan trong nhat (Delta lon nhat) -- da kiem bang artifact")

    def lines(ranks):
        res = []
        for g in GROUPS_ORDER:
            ss = [s for s in sorted(ranks) if g in ranks[s]]
            if len(ss) < 2:
                continue
            pts = "".join("(%d,%d)" % (i + 1, int(ranks[s][g])) for i, s in enumerate(ss))
            res.append(r"\addplot+[thick, mark size=1.9pt] coordinates {%s}; "
                       r"\node[font=\tiny, anchor=west] at (axis cs:%d,%d) {\texttt{%s}};"
                       % (pts, len(ss) + 0.05, int(ranks[ss[-1]][g]), g))
        return "\n".join(res)

    axis_common = (r"xmin=0.4, xmax=4.4, clip=false, xtick={1,2,3}, "
                   r"xticklabels={seed 42, seed 43, seed 44}, ymin=9.7, ymax=0.3, ytick={1,...,9}, "
                   r"grid=major, grid style={black!10}, tick label style={font=\footnotesize}")
    return fill(r"""
\begin{figure}[p]\centering
@@SRC@@
\begin{tikzpicture}
\begin{axis}[width=0.42\textwidth, height=7.8cm, name=A, @@COMMON@@,
  ylabel={Hạng utility (1 = quan trọng nhất)},
  title={Utility: hạng \emph{trôi} ($\rho=0{,}800$)},
  title style={font=\scriptsize}]
@@UTIL@@
\end{axis}
\begin{axis}[at={(A.south east)}, anchor=south west, xshift=1.1cm,
  width=0.42\textwidth, height=7.8cm, @@COMMON@@,
  ylabel={Hạng reliance},
  title={Reliance: hạng \emph{bất biến} ($\rho=1{,}000$)},
  title style={font=\scriptsize}]
@@RELI@@
\end{axis}
\end{tikzpicture}
\caption{Hai loại tái lập khác nhau hẳn nhau. \textbf{Bên phải}: hạng \emph{reliance} giống hệt nhau trên cả ba split ($\rho=1{,}000$, độ lệch hạng lớn nhất $=0$; exact $p$ tính bằng cách liệt kê toàn bộ $9!=362.880$ thứ tự $=5{,}5\times10^{-6}$). \textbf{Bên trái}: hạng \emph{utility} dịch chuyển ($\rho=0{,}800$, exact $p=0{,}0138$; \texttt{dlllist} đi từ hạng 7 lên hạng~3, $q_{\mathrm{BH}}$ của \texttt{ldrmodules} lật từ $0{,}1343$ xuống $0{,}0224$). Mỗi đường liền nét là một nhóm; dốc của nó chính là độ bất ổn của estimand tương ứng.}
\end{figure}
""", src=_src(*srcs), common=axis_common, util=lines(util), reli=lines(reli))


def fig_method_heatmap() -> str:
    path = RESULTS / "group_experiments" / comparison_dir() / "group_ranks.csv"
    if not path.exists():
        return prov_missing([str(path)])
    df = pd.read_csv(path, index_col=0)
    methods = [c for c in df.columns if c.endswith("_rank")]
    labels = {
        "ablation_rank": "Utility (retrain)",
        "permutation_rank": "Reliance (permute)",
        "shapley_rank": "Attribution (Shapley)",
        "cfi_rank": "Conditional (CFI)",
        "featurewise_rank": "Feature-wise (baseline)",
    }
    groups = [g for g in GROUPS_ORDER if g in df.index] + [g for g in df.index if g not in GROUPS_ORDER]
    cells = []
    for gi, g in enumerate(groups):
        for mi, m in enumerate(methods):
            v = df.loc[g, m]
            if pd.isna(v):
                colour, text = "fill=black!8", "--"
            else:
                frac = (float(v) - 1.0) / max(1, (len(groups) - 1))
                colour = r"fill=blue!%.0f!red!%.0f!white" % (14 + 40 * (1 - frac), 62 * frac)
                text = str(int(float(v)))
            cells.append(r"\node[hm,%s] at (%.2f,%.1f) {\textbf{\scriptsize %s}};"
                         % (colour, 1.0 + 1.6 * mi, len(groups) - gi, text))
    top = len(groups) + 0.62
    xlab = "\n".join(r"\node[hlab] at (%.2f,%.2f) {%s};"
                     % (1.0 + 1.6 * i, top, labels.get(m, m.replace("_rank", "")))
                     for i, m in enumerate(methods))
    ylab = "\n".join(r"\node[ylab] at (0.14,%.1f) {\texttt{\tiny %s}};"
                     % (len(groups) - i, g) for i, g in enumerate(groups))
    return fill(r"""
\begin{figure}[t]\centering
@@SRC@@
\begin{tikzpicture}[
  hm/.style={minimum width=1.5cm, minimum height=0.74cm, draw=black!25, inner sep=1pt},
  hlab/.style={align=center, font=\tiny, anchor=south, text width=1.58cm, inner sep=1pt},
  ylab/.style={anchor=east, font=\scriptsize},
]
@@XLAB@@
@@YLAB@@
@@CELLS@@
\end{tikzpicture}
\caption{Hạng nhóm theo năm phép đo (1 = quan trọng nhất; ô càng đỏ là hạng càng xấu). Năm cột \emph{không} phải năm cách đo cùng một đại lượng, nên thứ tự lệch nhau ở \texttt{pslist}, \texttt{svcscan} và \texttt{callbacks} là nội dung khoa học, không phải nhiễu. Cột CFI chỉ có số ở seed~42: bốn attempt CFI đa seed đều không chạy được trên máy 4{,}9\,GB.}
\end{figure}
""", src=_src("results/group_experiments/%s/group_ranks.csv" % comparison_dir(),
               "seed 42; utility va reliance chay 200 doi chung"),
            xlab=xlab, ylab=ylab, cells="\n".join(cells))


def comparison_dir() -> str:
    for name in ("comparison_v3", "comparison_v2", "comparison_v1"):
        if (RESULTS / "group_experiments" / name / "group_ranks.csv").exists():
            return name
    return "comparison_v3"


def fig_control_floor() -> str:
    return r"""
\begin{figure}[t]\centering
""" + _src("p thuc nghiem = (#{doi chung >= quan sat}+1)/(n+1)",
           "san p do n quyet dinh, khong do effect size") + r"""
\begin{tikzpicture}
\begin{axis}[
  width=0.86\textwidth, height=6.2cm,
  xlabel={Số đối chứng khớp độ rộng trên mỗi nhóm ($n$)},
  ylabel={Sàn $p$ thực nghiệm $=1/(n+1)$},
  xmin=0, xmax=260, ymin=0, ymax=0.30,
  grid=major, grid style={black!10}, tick label style={font=\footnotesize},
]
\addplot+[domain=1:260, samples=250, thick, no marks, blue!70!black] {1/(x+1)};
\node[font=\scriptsize, anchor=south east, text=blue!70!black] at (axis cs:34,0.150) {$1/(n+1)$};
\addplot+[only marks, mark=*, mark size=3pt, red!70!black] coordinates {(20,0.047619)};
\addplot+[only marks, mark=square*, mark size=3pt, teal!70!black] coordinates {(200,0.004975)};
\draw[dashed, black!45] (axis cs:0,0.05) -- (axis cs:260,0.05);
\node[font=\scriptsize, anchor=south west] at (axis cs:3,0.052) {$\alpha=0{,}05$};
\node[font=\scriptsize, anchor=north west, text=red!70!black, align=left, text width=7.2cm]
  at (axis cs:55,0.295) {$n=20$: sàn $p=0{,}0476$ --- sát ngưỡng, nên q\textsubscript{BH} của hạng nhất
   không bao giờ xuống dưới $0{,}0476$; mọi tuyên bố ở $n=20$ chỉ là \emph{pilot}};
\node[font=\scriptsize, anchor=north west, text=teal!70!black, align=left, text width=6.0cm]
  at (axis cs:55,0.185) {$n=200$: sàn $p=0{,}004975$; điều kiện cần $n\geq k/\alpha-1=179$ cho $q<0{,}05$ với $k=9$};
\end{axis}
\end{tikzpicture}
\caption{Vì sao bằng chứng chính phải chạy 200 đối chứng: thêm seed giảm variance nhưng \emph{không} hạ được sàn~$p$; chỉ thêm đối chứng mới hạ được. Đây là lý do các bảng $n=20$ trong dự án bị giáng xuống pilot và toàn bộ tuyên bố chính đặt trên $n=200$.}
\end{figure}
"""


def fig_width_confound() -> str:
    """Do RONG nhom = 55 - (so cot con lai).

    Cot `summary.csv:n_features` la so cot CON LAI sau khi bo nhom, khong phai
    do rong nhom; dung no truc tiep se dao dau rho (xem bao cao §5.1c).
    """
    a = read_summary("ablation_v4", "summary.csv")
    r = read_summary("importance_v3", "permutation_summary.csv")
    if a is None or r is None:
        return prov_missing(["thieu ablation_v4/summary.csv hoac importance_v3/permutation_summary.csv"])
    df = a[["group", "n_features"]].merge(r[["group", "delta_log_loss_mean"]], on="group")
    df = df.set_index("group").reindex(GROUPS_ORDER).dropna().reset_index()
    df["width"] = TOTAL_FEATURES - df["n_features"]
    rho = df["width"].corr(df["delta_log_loss_mean"], method="spearman")
    coords = "".join("(%d,%s)" % (int(w), _g(d)) for w, d in
                     zip(df["width"], df["delta_log_loss_mean"]))
    # oreach bi pgfplots ghi de trong moi truong axis -> sinh node truc tiep
    nodes = "\n".join(
        r"\node[font=\tiny, anchor=south west] at (axis cs:%d,%s) {\texttt{%s}};"
        % (int(row["width"]), _g(row["delta_log_loss_mean"]), row["group"])
        for _, row in df.iterrows())
    table = "\n".join(
        r"  \texttt{%-11s} & %2d cot & $\Delta=%s$ \\" % (row["group"], int(row["width"]),
                                                             _g(row["delta_log_loss_mean"]))
        for _, row in df.sort_values("width").iterrows())
    return fill(r"""
\begin{figure}[t]\centering
@@SRC@@
\begin{tikzpicture}
\begin{axis}[
  width=0.80\textwidth, height=6.8cm,
  xlabel={Độ rộng nhóm (số cột Volatility trong 55 feature)},
  ylabel={$\Delta$ reliance (log-loss)},
  xmin=0, xmax=15.5, ymin=-0.10, ymax=1.35, clip=false,
  grid=major, grid style={black!10}, tick label style={font=\footnotesize},
]
\addplot+[only marks, mark=*, mark size=2.8pt, blue!70!black!85] coordinates {@@COORDS@@};
@@NODES@@
\end{axis}
\end{tikzpicture}
\begin{center}\small
\begin{tabular}{@{}lrl@{}}
\toprule
Nhóm & Độ rộng & $\Delta$ reliance \\
\midrule
@@TABLE@@
\bottomrule
\end{tabular}
\end{center}
\caption{Độ rộng nhóm không giải thích được thứ tự reliance: $\rho(\text{độ rộng}, \Delta)$ $=@@RHO@@$ ($p=0{,}433$, cùng giá trị ở cả ba split). Ba điểm quyết định nằm ngay trên hình: \texttt{psxview} là nhóm \emph{rộng nhất} (14 cột) nhưng $\Delta$ chỉ 0,0228 (hạng 8/9); \texttt{handles} 13 cột -- gần như cùng độ rộng -- có $\Delta$ 1,1684, gấp $\sim$51 lần; \texttt{modules} \emph{hẹp nhất} (1 cột) có $\Delta$ nhỏ nhất. Trục $x$ của hình này không dự đoán được trục $y$.}
\end{figure}
""", src=_src("delta = results/group_experiments/importance_v3/permutation_summary.csv",
               "do rong = 55 - cot n_features cua results/group_experiments/ablation_v4/summary.csv",
               "n_features cua summary.csv la so cot CON LAI sau khi bo nhom",
               "rho Spearman do rong that = %.3f" % rho),
            coords=coords, nodes=nodes, table=table, rho="%+.2f" % rho)


def fig_group_rank_stability() -> str:
    path = RESULTS / "stability/stability_v1/stability_groups.csv"
    if not path.exists():
        return prov_missing(["results/stability/stability_v1/stability_groups.csv thieu"])
    df = pd.read_csv(path).set_index("group").reindex(GROUPS_ORDER).dropna().reset_index()
    pts = " ".join("(%d,%s)+- (%s,%s)" % (i + 1, _g(row["rank_mean"]),
                                          _g(row["rank_mean"] - row["ci_lo"]),
                                          _g(row["ci_hi"] - row["rank_mean"]))
                   for i, row in df.iterrows())
    xticks = ",".join(str(i + 1) for i in range(len(df)))
    return fill(r"""
\begin{figure}[t]\centering
@@SRC@@
\begin{tikzpicture}
\begin{axis}[
  width=\textwidth, height=6.8cm,
  ylabel={Hạng trung bình (thấp = ổn định hơn)},
  xlabel={Nhóm (sắp theo hạng trung bình; trái = quan trọng nhất)},
  xmin=0.4, xmax=@@XMAX@@, xtick={@@XTICK@@}, xticklabels={@@XTLAB@@},
  ymin=0, ymax=10,
  grid=major, grid style={black!10}, tick label style={font=\footnotesize},
]
\addplot+[only marks, mark=*, mark size=2.6pt, blue!70!black,
          error bars/.cd, y dir=both, y explicit]
  coordinates {@@PTS@@};
\end{axis}
\end{tikzpicture}
\caption{Độ ổn định của \emph{importance} feature-wise qua các lần fit (thay cho hình PNG cũ \texttt{results/stability/stability\_v1/figures/}). Râu là khoảng tin cậy 95\,\%. Đây là stability ở tầng feature; tái lập ở tầng \emph{nhóm} xem hình hạng trôi/bất biến.}
\end{figure}
""", src=_src("results/stability/stability_v1/stability_groups.csv",
               "bootstrap theo don vi doc lap; rau = khoang tin cậy 95 phan tram"),
            xmax="%d.6" % len(df), xtick=xticks,
            xtlab=",".join(r"\texttt{%s}" % g for g in df["group"]), pts=pts)


def prov_missing(parts: list[str]) -> str:
    return "%% MISSING HINH: " + "; ".join(parts) + "\n"


BOUND_RE = re.compile(r"^\s+(\S+) ctrl (\d+)/(\d+) \((\d+)s elapsed\)")


def _group_boundaries(log_path: Path) -> list:
    """(elapsed_s, so nhom tich luy) tai moi boundary TRON nhom trong log ablation."""
    out, seen, last_grp = [], 0, None
    if not log_path.exists():
        return out
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BOUND_RE.match(line)
        if m and int(m.group(2)) >= int(m.group(3)) and m.group(1) != last_grp:
            last_grp = m.group(1)
            seen += 1
            out.append((int(m.group(4)), seen))
    return out


def fig_serialized_chain() -> str:
    """Tuoi ETA cua job ablation dang chay tu chinh log, khong tu tri nho."""
    d44 = RESULTS / "group_experiments" / "ablation_xgb_s44.log"
    d43 = RESULTS / "group_experiments" / "ablation_xgb_s43.log"
    b44, b43 = _group_boundaries(d44), _group_boundaries(d43)
    if b44 and b43:
        elapsed, ndone = b44[-1]
        ref = next((e for e, k in b43 if k == ndone), None)
        total43 = b43[-1][0]
        if ref and total43:
            # Chi ngoai suy bang ti so tich luy tai boundary tron nhom. Ti so nay
            # troi co he thong trong run nay nen moi uoc tinh tu boundary som deu
            # sai mot phia; thu tu boundary gan nhat la duyet duoc duy nhất.
            ratio = elapsed / ref
            left = max(0.0, (total43 * ratio - elapsed)) / 3600.0
            abl_box = ("%d/9 nhom, $\\approx$ %s\\,s tich luy\\\\ "
                       "ti le s43 = %.3f $\\cdot$ con $\\approx$ %.1f\\,h"
                       % (ndone, format(elapsed, ",d").replace(",", "."), ratio, left))
        else:
            abl_box = "%d/9 nhom, chura du boundary de ngoai suy" % ndone
    else:
        abl_box = "chura co log"
    return r"""
\begin{figure}[t]\centering
""" + _src("scripts/chain_after.sh", "watchdog = results/group_experiments/watchdog.log",
           "thoi gian do tu log/artifact cua tung run") + r"""
\begin{tikzpicture}[
  job/.style={draw=black!55, rounded corners=2pt, align=center, font=\scriptsize,
              inner sep=4pt, minimum width=2.85cm, fill=black!4},
  gate/.style={draw=orange!70!black, align=center, font=\scriptsize, inner sep=3pt,
               regular polygon=regular polygon sides=4, minimum size=1.0cm, fill=orange!8},
  arr/.style={-{Latex[length=2.4mm]}, thick, draw=black!60},
]
\node[job, fill=teal!10] (p) at (0,0) {Permutation seed 44\\ \texttt{importance\_v3\_s44}\\ 11.342\,s $\approx$ 3,15\,h\\ \emph{ĐÃ CHẠY, verifier PASS}};
\node[job] (a) at (4.2,0) {Ablation seed 44\\ \texttt{ablation\_xgb\_s44}\\ @@ABL@@\\ \emph{đang chạy}};
\node[job] (l) at (8.4,0) {Ablation LogReg\\ \texttt{ablation\_logreg\_v4}\\ $\approx$ 2,7--3,0\,h};
\node[job] (i) at (12.2,0) {Permutation LogReg\\ \texttt{importance\_v3\_logreg}\\ $\approx$ 7--12\,phút};
\draw[arr] (p) -- (a); \draw[arr] (a) -- (l); \draw[arr] (l) -- (i);
\node[gate] (g1) at (2.1,-1.5) {\,RAM};
\node[gate] (g2) at (6.3,-1.5) {\,RAM};
\node[gate] (g3) at (10.3,-1.5) {\,RAM};
\draw[black!45, dotted] (p.south) -- (g1.135);
\draw[black!45, dotted] (a.south) -- (g1.45) (a.south) -- (g2.135);
\draw[black!45, dotted] (l.south) -- (g2.45) (l.south) -- (g3.135);
\draw[black!45, dotted] (i.south) -- (g3.45);
\node[align=center, font=\scriptsize, text=black!72, anchor=north] at (6.1,-2.35)
  {\texttt{chain\_after.sh} chờ PID của job trước \emph{thoát hẳn}, và chỉ khởi động job sau nếu\\
   \texttt{MemAvailable} $\geq$ 1.500\,MB --- một tiến trình Python nặng duy nhất tại một thời điểm.\\
   Cơ chế này là phản ứng trực tiếp trước hai lần watchdog gửi SIGTERM tới tiến trình R\\
   (\texttt{avail=157MB, rss=3611MB}; \texttt{avail=229MB, rss=3435MB}) khi CFI bị OOM.};
\node[align=left, font=\scriptsize, text width=11.5cm, draw=red!60!black, dashed, rounded corners=2pt, inner sep=5pt,
      text=red!70!black, anchor=north west] at (-0.3,-4.15)
  {HistGB 200 đối chứng: \textbf{không xếp hàng} --- quyết định có chủ đích. Chi phí đo từ pilot 20
   đối chứng: permutation $\approx$ 10,8\,h (4.925,1\,s ở 20 đối chứng), ablation $\approx$ 21--23\,h
   (925\,s/nhóm) --- gấp $\sim$3 lần toàn bộ phần còn lại mà không đổi được kết luận nào đã ghi,
   chỉ thêm một dòng model ở sàn $p$ thấp.};
\end{tikzpicture}
\caption{Chuỗi thí nghiệm nối tiếp trên máy 4{,}9\,GB và cổng bộ nhớ giữa các job. Mọi số giờ trên hình lấy từ log/artifact của chính run đó.}
\end{figure}
""".replace("@@ABL@@", abl_box)


FIGURES = [
    ("fig_estimand_pipeline", fig_estimand_pipeline),
    ("fig_reliance_splits", lambda: fig_reliance_splits(
        {"split42": "importance_v3", "split43": "importance_v3_s43", "split44": "importance_v3_s44"})),
    ("fig_rank_drift", fig_rank_drift),
    ("fig_method_heatmap", fig_method_heatmap),
    ("fig_control_floor", fig_control_floor),
    ("fig_width_confound", fig_width_confound),
    ("fig_group_rank_stability", fig_group_rank_stability),
    ("fig_serialized_chain", fig_serialized_chain),
]


def write_figures(figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for fname, fn in FIGURES:
        try:
            tex = fn()
        except Exception as exc:  # noqa: BLE001
            tex = "%% MISSING HINH (%s): %s\n" % (fname, exc)
        (figure_dir / ("%s.tex" % fname)).write_text(tex, encoding="utf-8")
        names.append(fname)
    return names


# --------------------------------------------------------------------------- #
# Preamble + main.tex
# --------------------------------------------------------------------------- #
PREAMBLE = r"""%% ===================================================================
%%  RAM-EVID --- bao cao LaTeX cho Overleaf
%%  Tieng Viet UTF-8.  Compiler: pdfLaTeX (mac dinh Overleaf).
%%  Tiang Viet can encoding T5 (package `vietnam`) -- Overleaf co san.  May cuc bo
%%  thieu T5 thi block \IfFileExists duoi day tu fallback ve T1 va in canh cao,
%%  nen du an van bien dich duoc de kiem tra cau truc.
%%  main.tex \input sec/*.tex va figures/*.tex.  TOAN BO sec/ va figures/ do
%%  python/make_latex_report.py sinh: SUA MARKDOWN/ARTIFACT ROI CHAY LAI,
%%  DUNG SUA TAY FILE SINH.
%% ===================================================================
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\IfFileExists{t5enc.def}{%
  \usepackage[T5]{fontenc}%
  \usepackage{vietnam}%
}{%
  \usepackage[T1]{fontenc}%
  \PackageWarning{rantevid}{Khong tim thay T5/vietnam: bien dich voi T1, dau tieng Viet co the sai. Dung Overleaf (co T5) de ban dung.}%
}
\usepackage{lmodern}
\usepackage{textcomp}
\usepackage[a4paper,top=2.4cm,bottom=2.4cm,left=2.4cm,right=2.4cm,headheight=15pt,footskip=26pt]{geometry}
\usepackage{microtype}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{graphicx}
\usepackage{adjustbox}
\IfFileExists{xurl.sty}{\usepackage{xurl}}{\usepackage{url}}
\usepackage{multirow}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{makecell}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{needspace}
\usepackage{caption}
\usepackage{listings}
\usepackage{pgfplots}
\usepackage{tikz}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta,positioning,calc,backgrounds,shapes.geometric}
\IfFileExists{lastpage.sty}{\usepackage{lastpage}\newcommand{\rantevidlast}{/\pageref{LastPage}}}{\newcommand{\rantevidlast}{}}
\usepackage[colorlinks=true,linkcolor=blue!55!black,urlcolor=teal!60!black,
            citecolor=blue!55!black,bookmarks=false]{hyperref}
\usepackage{fancyhdr}

\setlist[itemize]{leftmargin=1.4em,topsep=2pt,itemsep=1pt,parsep=0pt}
\setlist[enumerate]{leftmargin=2.0em,topsep=2pt,itemsep=1pt,parsep=0pt}
\setlength{\parskip}{4pt}
\setlength{\emergencystretch}{3em}
\hbadness=8000 \vbadness=8000 \relpenalty=30 \binoppenalty=50
\setlength{\parindent}{0pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{2}
\renewcommand{\arraystretch}{1.18}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\captionsetup[table]{position=top}
\titleformat{\section}{\large\bfseries}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\small\bfseries\itshape}{\thesubsubsection}{0.5em}{}

\lstdefinestyle{base}{
  basicstyle=\ttfamily\scriptsize,
  columns=fullflexible, keepspaces=true, breaklines=true,
  breakatwhitespace=false, showstringspaces=false,
  frame=single, rulecolor=\color{black!28}, backgroundcolor=\color{black!2.5},
  xleftmargin=0.6em, xrightmargin=0.3em, aboveskip=6pt, belowskip=6pt,
  literate={->}{{$\rightarrow$}}2 {>=}{{$\geq$}}2 {<=}{{$\leq$}}2
            {|}{{\textbar}}1 {~}{{\textasciitilde}}1
}
\lstdefinestyle{shell}{style=base,
  morekeywords={python,python3,nohup,tail,grep,sed,awk,cd,git,timeout,free,find,mkdir,cat,head,wc,sort,uniq,cut,du,ps,ls,mvn,cp,latexmk}
}
\lstnewenvironment{codeblock}{\lstset{style=base}}{}

\newcommand{\pend}[1]{\textcolor{red!65!black}{\textit{[chưa có artifact: #1]}}}
\newcommand{\labdone}{\textcolor[rgb]{0.10,0.42,0.18}{\checkmark\,\textsc{đã chạy}}}
\newcommand{\labrun}{\textcolor[rgb]{0.72,0.42,0.03}{$\triangleright$\,\textsc{đang chạy}}}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\itshape RAM-EVID --- độ tin cậy của phép đo tầm quan trọng nhóm đặc trưng}
\fancyhead[R]{\footnotesize\thepage\rantevidlast}
\fancyfoot[C]{\footnotesize\itshape Mọi số liệu truy vết được tới artifact trên đĩa --- xem nguồn ghi trong từng bảng/hình.}
\renewcommand{\headrulewidth}{0.4pt}

\hypersetup{
  pdftitle={RAM-EVID: do tin cay cua cac phep do tam quan trong nhom dac trung trong phan tich forensics bo nho},
  pdfauthor={RAM-EVID},
  pdfsubject={XAI reliability study on CIC-MalMem-2022},
}
"""

TITLE_BLOCK = r"""\begin{document}
\begin{titlepage}
\centering
{\large\sffamily RAM-EVID}\\[0.8cm]
{\LARGE\bfseries Độ tin cậy của các phép đo tầm quan trọng nhóm đặc trưng\\[3pt]
trong phân tích forensics bộ nhớ}\\[6pt]
{\large\itshape So sánh năm cách đo trên CIC-MalMem-2022\par}
\vspace{0.9cm}
{\small
\begin{tabular}{@{}>{\bfseries}l@{\hspace{1em}}l@{}}
Dataset & \texttt{data/Obfuscated-MalMem2022.csv} (SHA-256 \texttt{624ecb6c...3840c4})\\
Task & \texttt{family16}; 58.596 dòng gốc, 58.008 vector sau loại trùng, 55 feature\\
Nhóm đặc trưng & 9 nhóm theo plugin Volatility\\
Model chính & XGBoost (đối chiếu: LogReg, HistGB)\\
Split khoá & \texttt{source\_aware\_v1}; seed 42, 43, 44\\
Đối chứng & 200 per nhóm, \emph{khớp độ rộng}; sàn $p=1/201=0{,}004975$\\
Hiệu chỉnh & Benjamini--Hochberg trên $k=9$ nhóm, mỗi phép đo riêng\\
Môi trường & WSL2, trần 4{,}9\,GB cho toàn VM\\
\end{tabular}\par}
\vfill
{\small\itshape Sinh ngày \the\day/\the\month/\the\year\ bởi \texttt{python/make\_latex\_report.py}.\\
Bảng: lấy từ các báo cáo markdown đã được \texttt{scripts/verify\_replication.py} đối chiếu máy với artifact.\\
Hình: TikZ/pgfplots đọc trực tiếp artifact, nguồn ghi trong \texttt{\textbackslash typeout} và chú thích.\par}
\end{titlepage}
\tableofcontents
\newpage
"""


def build(out: Path, reports: list[str], do_check: bool) -> int:
    sec, fig = out / "sec", out / "figures"
    shutil.rmtree(sec, ignore_errors=True)
    shutil.rmtree(fig, ignore_errors=True)
    sec.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    generated = write_figures(fig)
    inputs, missing = [], []
    for rel in reports:
        src = ROOT / rel
        if not src.exists():
            missing.append(rel)
            continue
        stem = Path(rel).stem
        tex = md_to_tex(src.read_text(encoding="utf-8"))
        (sec / ("%s.tex" % stem)).write_text(
            "%% %s --- sinh tu %s boi python/make_latex_report.py luc %s\n" % (
                stem, rel, dt.datetime.now().strftime("%Y-%m-%d %H:%M")) + tex,
            encoding="utf-8")
        inputs.append("sec/%s" % stem)

    (sec / "zz_figures.tex").write_text(
        "\\section{Hình minh họa sinh trực tiếp từ artifact}\n"
        "%% Moi hinh mang theo dong \\typeout chi danh artifact cua no.\n" +
        "\n".join(r"\input{figures/%s}" % n for n in generated) + "\n",
        encoding="utf-8")

    body = "\n".join(r"\input{%s}" % p for p in inputs)
    (out / "main.tex").write_text(
        PREAMBLE + "\n" + TITLE_BLOCK + body
        + "\n\\input{sec/zz_figures}\n\n"
        + "\\end{document}\n",
        encoding="utf-8")

    print("latex/: %d báo cáo -> sec/*.tex, %d hình, main.tex" % (len(inputs), len(generated)))
    for name in generated:
        head = (fig / ("%s.tex" % name)).read_text(encoding="utf-8")
        flag = "MISSING" if "MISSING HINH" in head or "% MISSING" in head else "ok"
        print("   %-28s %s" % (name, flag))
    if missing:
        print("  cảnh cáo, bỏ qua vì không tồn tại:", ", ".join(missing))

    # Chuỗi Python khong raw bien \a thanh BEL va \t thanh TAB, lam hong
    # \arraybackslash va \textwidth ben trong file .tex. Chet o day tot hon
    # de LaTeX bao "Unicode character not set up".
    bad = []
    for f in sorted(out.rglob("*.tex")):
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if CTRL_RE.search(line):
                bad.append("%s:%d" % (f.name, i))
    if bad:
        print("  LOI SINH, ky tu dieu khong trong .tex:", ", ".join(bad[:10]))
        return 1
    return run_check(out) if do_check else 0


def asciiize(text: str) -> str:
    fixed = {"ơ": "o", "ư": "u", "đ": "d", "Đ": "D"}
    out = []
    for ch in text:
        if ch in fixed:
            out.append(fixed[ch])
            continue
        d = unicodedata.normalize("NFD", ch)
        base = "".join(c for c in d if not unicodedata.combining(c))
        out.append(base if base else ch)
    return "".join(out)


def run_check(out: Path) -> int:
    check = Path("/tmp/latex_check")
    if check.exists():
        shutil.rmtree(check)
    check.mkdir(parents=True)
    n = 0
    for p in sorted(out.rglob("*.tex")):
        dst = check / p.relative_to(out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(asciiize(p.read_text(encoding="utf-8")), encoding="utf-8")
        n += 1
    print("[check] %d file .tex -> ban ASCII tai %s (chi de kiem cau truc, khong phai ban nop)" % (n, check))
    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    rc, prev = 0, None
    for _ in range(4):
        # Dung vong lap khi (so trang xuat ra, so trang LastPage) lap lai nguyen ven
        # qua hai lan bien dich. Khong so sanh hai gia tri nay bang nhau: trang bi
        # a khong dem so nen lech nhau mot, do la quy uoc hop le cua article.
        rc = subprocess.run(cmd, cwd=check, stdout=subprocess.DEVNULL,
                            stderr=subprocess.STDOUT).returncode
        if rc != 0:
            break
        log = (check / "main.log").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Output written on main.pdf \((\d+) pages", log)
        aux = (check / "main.aux").read_text(encoding="utf-8", errors="replace")
        lp = re.search(r"LastPage\}\{\{\}\{(\d+)\}", aux)
        cur = (m.group(1) if m else None, lp.group(1) if lp else None)
        if cur == prev and cur[0] is not None:
            break
        prev = cur
    log = (check / "main.log").read_text(encoding="utf-8", errors="replace")
    if rc != 0:
        print("[check] LOI BIEN DICH (rc=%d)" % rc)
        ctx = log.splitlines()
        for i, line in enumerate(ctx):
            if line.startswith("!"):
                print("   ", "\n    ".join(ctx[max(0, i - 3):i + 3])[:600])
                break
        return 1
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    warns = [l for l in log.splitlines() if l.startswith("Warning")]
    over = [l for l in log.splitlines() if "Overfull" in l]
    print("[check] OK --- main.pdf %s trang; %d canh cao; %d cho tran dong" % (
        pages.group(1) if pages else "?", len(warns), len(over)))
    for l in warns[:6]:
        print("   W:", l[:160])
    if over:
        print("   Overfull dau tien:", over[0][:160])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sinh du an LaTeX/Overleaf tu bao cao markdown")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--reports", default=",".join(REPORTS))
    ap.add_argument("--check", action="store_true", help="bien dich thu bang pdflatex (ban ASCII)")
    a = ap.parse_args(argv)
    reports = [r.strip() for r in a.reports.split(",") if r.strip()]
    return build(Path(a.out), reports, a.check)


if __name__ == "__main__":
    sys.exit(main())
