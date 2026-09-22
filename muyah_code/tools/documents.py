"""Documents as text for the Read tool: PDF, Word, Excel, PowerPoint, OpenDocument, notebooks, e-books, RTF,
emails, archives (listed, never unpacked) and SQLite databases (the schema, opened read-only).

Office files (.docx/.xlsx/.pptx) and OpenDocument files are ZIP archives of XML, read here with the standard
library only: headings, list items and tables survive as markdown, spreadsheets come out as rows per sheet,
slides one by one. PDFs use pypdf (text per page; `pages` picks a range, like Claude Code's Read). Old binary
Office files (.doc/.xls/.ppt) are converted with LibreOffice when it is installed.

Files come from anywhere, so parsing is defensive: XML that declares a DTD or entities is refused (Office
XML never does; it is how billion-laughs and external-entity attacks work), each archive member is capped
at MAX_PART bytes uncompressed (ZIP bombs), and archives are only listed.

Creating these files is done by writing a small Python script (see the bundled `documents` skill): a real
.docx or .xlsx needs a real library, not text written with a .docx name.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_R = "{http://schemas.openxmlformats.org/package/2006/relationships}"
TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
TABLE_NS = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"

ARCHIVE_EXT = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip", ".whl", ".jar", ".apk", ".nupkg")
DOCUMENT_EXT = {".pdf", ".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm", ".odt", ".ods", ".odp", ".ipynb",
                ".doc", ".xls", ".ppt", ".rtf", ".epub", ".eml", ".db", ".sqlite", ".sqlite3"}
LEGACY = {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}
MAX_PART = 64 * 1024 * 1024  # uncompressed bytes allowed for one archive member (a ZIP bomb stops here)
MAX_LISTED = 500             # archive entries listed
MAX_ROWS = 2000              # per sheet: a spreadsheet can be huge; the rest is counted, not shown
MAX_PDF_PAGES = 20           # per read, as Claude Code does


class DocumentError(Exception):
    """The file is a document we know, but it could not be read (damaged, encrypted, missing library)."""


def is_document(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in DOCUMENT_EXT or name.endswith(ARCHIVE_EXT)


def _xml(data: bytes):
    """Parse XML from an untrusted file: no DTDs, no entity declarations."""
    head = data[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in data.upper():
        raise DocumentError("the file contains XML entity declarations, which documents never need; "
                            "it was not parsed (it may be crafted to attack XML readers)")
    try:
        from defusedxml.ElementTree import fromstring   # hardened parser (a dependency); the check above stays
    except ImportError:
        fromstring = ET.fromstring
    return fromstring(data)


def _read(z: zipfile.ZipFile, name: str) -> bytes:
    info = z.getinfo(name)
    if info.file_size > MAX_PART:
        raise DocumentError(f"{name} inside the file would unpack to {info.file_size // 1_000_000} MB; "
                            "that is too big to read safely")
    return z.read(info)


def extract(path: Path, pages: str | None = None) -> tuple[str, str]:
    """(text, what it is) for a document. Raises DocumentError with a reason the model can act on."""
    ext = path.suffix.lower()
    name = path.name.lower()
    try:
        if name.endswith(ARCHIVE_EXT) and not name.endswith((".docx", ".xlsx", ".pptx")):
            return _archive(path)
        if ext in (".db", ".sqlite", ".sqlite3"):
            return _sqlite(path)
        if ext == ".eml":
            return _email(path)
        if ext == ".rtf":
            return _rtf(path), "RTF document"
        if ext == ".pdf":
            return _pdf(path, pages)
        if ext == ".ipynb":
            return _notebook(path)
        if ext in LEGACY:
            return _legacy(path, LEGACY[ext])
        if not zipfile.is_zipfile(path):
            raise DocumentError(f"{path.name} is not a valid {ext} file (it is not the ZIP archive Office uses)")
        with zipfile.ZipFile(path) as z:
            if ext in (".docx", ".docm"):
                return _docx(z), "Word document"
            if ext in (".xlsx", ".xlsm"):
                return _xlsx(z), "Excel workbook"
            if ext in (".pptx", ".pptm"):
                return _pptx(z), "PowerPoint presentation"
            if ext == ".epub":
                return _epub(z), "e-book"
            return _opendocument(z), {"odt": "OpenDocument text", "ods": "OpenDocument spreadsheet",
                                      "odp": "OpenDocument presentation"}[ext[1:]]
    except (KeyError, ET.ParseError, zipfile.BadZipFile, UnicodeDecodeError) as e:
        raise DocumentError(f"{path.name} looks damaged and could not be read ({e})") from e


# ---------------------------------------------------------------------------- Word

def _docx(z: zipfile.ZipFile) -> str:
    root = _xml(_read(z, "word/document.xml"))
    numbered = _numbered_styles(z)
    body = root.find(f"{W}body")
    out: list[str] = []
    for block in body if body is not None else []:
        if block.tag == f"{W}p":
            line = _docx_paragraph(block, numbered)
            if line is not None:
                out.append(line)
        elif block.tag == f"{W}tbl":
            out.extend(_docx_table(block))
            out.append("")
    return _tidy("\n".join(out))


def _numbered_styles(z: zipfile.ZipFile) -> set[str]:
    """Paragraph styles that are list styles (their paragraphs are list items even without numPr)."""
    try:
        styles = _xml(_read(z, "word/styles.xml"))
    except KeyError:
        return set()
    return {s.get(f"{W}styleId", "") for s in styles.iter(f"{W}style")
            if s.find(f"{W}pPr/{W}numPr") is not None}


def _docx_paragraph(p, numbered: set[str]) -> str | None:
    text = "".join(_docx_runs(p)).strip()
    style_el = p.find(f"{W}pPr/{W}pStyle")
    style = style_el.get(f"{W}val", "") if style_el is not None else ""
    if not text:
        return ""
    heading = re.match(r"(?i)heading\s*(\d)|title", style)
    if heading:
        level = int(heading.group(1)) if heading.group(1) else 1
        return "#" * min(6, level) + " " + text
    num = p.find(f"{W}pPr/{W}numPr")
    if num is not None or style in numbered or style.lower().startswith("list"):
        ilvl = num.find(f"{W}ilvl") if num is not None else None
        depth = int(ilvl.get(f"{W}val", "0")) if ilvl is not None else 0
        return "  " * depth + "- " + text
    return text


def _docx_runs(el):
    for node in el.iter():
        if node.tag == f"{W}t":
            yield node.text or ""
        elif node.tag == f"{W}tab":
            yield "\t"
        elif node.tag in (f"{W}br", f"{W}cr"):
            yield "\n"


def _docx_table(tbl) -> list[str]:
    rows = []
    for tr in tbl.iter(f"{W}tr"):
        cells = [" ".join("".join(_docx_runs(p)).strip() for p in tc.iter(f"{W}p")).strip().replace("|", "\\|")
                 for tc in tr.findall(f"{W}tc")]
        rows.append(cells)
    return _markdown_table(rows)


def _markdown_table(rows: list[list[str]]) -> list[str]:
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


# ---------------------------------------------------------------------------- Excel

def _xlsx(z: zipfile.ZipFile) -> str:
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in _xml(_read(z, "xl/sharedStrings.xml")).iter(f"{S}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{S}t")))
    book = _xml(_read(z, "xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in _xml(_read(z, "xl/_rels/workbook.xml.rels")).iter(f"{PKG_R}Relationship")}
    out: list[str] = []
    for sheet in book.iter(f"{S}sheet"):
        target = rels.get(sheet.get(f"{R}id"), "")
        name = "xl/" + target.lstrip("/").removeprefix("xl/")
        if name not in z.namelist():
            continue
        rows, total = _xlsx_rows(_xml(_read(z, name)), shared)
        out.append(f"## Sheet: {sheet.get('name')}" + (f" ({total} rows, first {MAX_ROWS} shown)" if total > len(rows) else ""))
        out.extend(",".join(_csv_cell(c) for c in r) for r in rows)
        out.append("")
    return _tidy("\n".join(out))


def _xlsx_rows(ws, shared: list[str]) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    total = 0
    for row in ws.iter(f"{S}row"):
        total += 1
        if len(rows) >= MAX_ROWS:
            continue
        cells: dict[int, str] = {}
        for c in row.findall(f"{S}c"):
            col = _column(c.get("r", "A1"))
            kind = c.get("t")
            v = c.find(f"{S}v")
            if kind == "inlineStr":
                value = "".join(t.text or "" for t in c.iter(f"{S}t"))
            elif v is None:
                f = c.find(f"{S}f")
                value = f"={f.text}" if f is not None and f.text else ""
            elif kind == "s":
                value = shared[int(v.text)] if v.text and v.text.isdigit() and int(v.text) < len(shared) else ""
            elif kind == "b":
                value = "TRUE" if v.text == "1" else "FALSE"
            else:
                value = v.text or ""
            cells[col] = value
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows, total


def _column(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]*", ref.upper()).group(0):
        n = n * 26 + (ord(ch) - 64)
    return max(0, n - 1)


def _csv_cell(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"' if any(ch in value for ch in ',"\n') else value


# ---------------------------------------------------------------------------- PowerPoint

def _pptx(z: zipfile.ZipFile) -> str:
    slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
    out: list[str] = []
    for i, name in enumerate(slides, 1):
        root = _xml(_read(z, name))
        paragraphs = ["".join(t.text or "" for t in p.iter(f"{A}t")).strip() for p in root.iter(f"{A}p")]
        paragraphs = [p for p in paragraphs if p]
        out.append(f"## Slide {i}" + (f": {paragraphs[0]}" if paragraphs else ""))
        out.extend(f"- {p}" for p in paragraphs[1:])
        notes = f"ppt/notesSlides/notesSlide{name.rsplit('slide', 1)[1]}"
        if notes in z.namelist():
            said = " ".join(t.text or "" for t in _xml(_read(z, notes)).iter(f"{A}t")).strip()
            if said:
                out.append(f"Notes: {said}")
        out.append("")
    return _tidy("\n".join(out))


# ---------------------------------------------------------------------------- OpenDocument

def _opendocument(z: zipfile.ZipFile) -> str:
    root = _xml(_read(z, "content.xml"))
    out: list[str] = []
    for el in root.iter():
        if el.tag == f"{TEXT_NS}h":
            level = int(el.get(f"{TEXT_NS}outline-level", "1"))
            out.append("#" * min(6, level) + " " + "".join(el.itertext()).strip())
        elif el.tag == f"{TEXT_NS}p":
            text = "".join(el.itertext()).strip()
            if text:
                out.append(text)
        elif el.tag == f"{TABLE_NS}table":
            out.append(f"## Table: {el.get(f'{TABLE_NS}name', '')}")
    return _tidy("\n".join(out))


# ---------------------------------------------------------------------------- PDF

def _pdf(path: Path, pages: str | None) -> tuple[str, str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as e:
        raise DocumentError('reading PDFs needs pypdf: pip install pypdf') from e
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError(f"{path.name} is password-protected")
        count = len(reader.pages)
        first, last = _page_range(pages, count)
        parts = []
        for n in range(first, last + 1):
            text = (reader.pages[n - 1].extract_text() or "").strip()
            parts.append(f"--- page {n} ---\n{text or '(no text on this page: it may be a scanned image)'}")
    except PdfReadError as e:
        raise DocumentError(f"{path.name} could not be read as a PDF ({e})") from e
    note = ""
    if first > 1 or last < count:
        note = f"\n\n[Pages {first}-{last} of {count}. Read more with pages=\"{last + 1}-{min(count, last + MAX_PDF_PAGES)}\".]" \
            if last < count else f"\n\n[Pages {first}-{last} of {count}.]"
    return "\n\n".join(parts) + note, f"PDF, {count} page{'s' if count != 1 else ''}"


def _page_range(pages: str | None, count: int) -> tuple[int, int]:
    if not pages:
        return 1, min(count, MAX_PDF_PAGES)
    m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", str(pages))
    if not m:
        raise DocumentError(f'pages must look like "3" or "1-5", not {pages!r}')
    first = max(1, int(m.group(1)))
    last = int(m.group(2) or first)
    if first > count:
        raise DocumentError(f"page {first} is past the end (the PDF has {count} pages)")
    last = min(count, last, first + MAX_PDF_PAGES - 1)
    return first, max(first, last)


# ---------------------------------------------------------------------------- notebooks

def _notebook(path: Path) -> tuple[str, str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise DocumentError(f"{path.name} is not a valid notebook ({e})") from e
    lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
    out: list[str] = []
    for i, cell in enumerate(nb.get("cells", []), 1):
        source = "".join(cell.get("source", []))
        kind = cell.get("cell_type", "code")
        out.append(f"## Cell {i} ({kind})")
        out.append(f"```{lang}\n{source}\n```" if kind == "code" else source)
        for o in cell.get("outputs", []):
            text = "".join(o.get("text", [])) or "".join(o.get("data", {}).get("text/plain", []))
            if o.get("output_type") == "error":
                text = f"{o.get('ename')}: {o.get('evalue')}"
            if text.strip():
                out.append("Output:\n" + text.strip()[:4000])
            elif "image/png" in o.get("data", {}):
                out.append("Output: [an image]")
        out.append("")
    return _tidy("\n".join(out)), f"Jupyter notebook, {len(nb.get('cells', []))} cells"


# ---------------------------------------------------------------------------- legacy formats

def _legacy(path: Path, to: str) -> tuple[str, str]:
    """Old binary .doc/.xls/.ppt (and .rtf): convert with LibreOffice, then read the modern file."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice") or _windows_soffice()
    if not soffice:
        raise DocumentError(f"{path.name} is an old {path.suffix} file. Reading it needs LibreOffice "
                            "(https://www.libreoffice.org); or ask the user to save it as "
                            f".{to} and read that")
    with tempfile.TemporaryDirectory(prefix="muyah-doc-") as tmp:
        try:
            subprocess.run([soffice, "--headless", "--convert-to", to, "--outdir", tmp, str(path)],
                           capture_output=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise DocumentError(f"LibreOffice could not convert {path.name} ({e})") from e
        converted = Path(tmp) / f"{path.stem}.{to}"
        if not converted.exists():
            raise DocumentError(f"LibreOffice could not convert {path.name}")
        text, kind = extract(converted)
    return text, f"{kind} (converted from {path.suffix})"


def _windows_soffice() -> str | None:
    for base in (Path("C:/Program Files/LibreOffice/program"), Path("C:/Program Files (x86)/LibreOffice/program")):
        exe = base / "soffice.exe"
        if exe.exists():
            return str(exe)
    return None


def _tidy(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


# ---------------------------------------------------------------------------- more formats

def _epub(z: zipfile.ZipFile) -> str:
    """Chapters in reading order (the OPF spine), as plain text."""
    container = _xml(_read(z, "META-INF/container.xml"))
    opf_path = next(el.get("full-path") for el in container.iter() if el.tag.endswith("rootfile"))
    opf = _xml(_read(z, opf_path))
    base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
    items = {el.get("id"): el.get("href") for el in opf.iter() if el.tag.endswith("item")}
    out: list[str] = []
    for ref in (el.get("idref") for el in opf.iter() if el.tag.endswith("itemref")):
        href = items.get(ref)
        if href and (base + href) in z.namelist():
            out.append(_html_text(_read(z, base + href).decode("utf-8", errors="replace")))
    return _tidy("\n\n".join(out))


def _html_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for level in range(1, 7):
        for h in soup.find_all(f"h{level}"):
            h.string = "#" * level + " " + h.get_text(" ", strip=True)
    return soup.get_text("\n", strip=True)


def _rtf(path: Path) -> str:
    """RTF to text without LibreOffice: drop control words and groups that are not text."""
    data = path.read_text(encoding="latin-1")
    skip = re.compile(r"\\\*|\\(fonttbl|colortbl|stylesheet|info|pict|header|footer|object)\b")
    out: list[str] = []
    depth_skip, depth, i = None, 0, 0
    while i < len(data):
        ch = data[i]
        if ch == "{":
            depth += 1
            if depth_skip is None and skip.match(data, i + 1):
                depth_skip = depth
        elif ch == "}":
            if depth_skip == depth:
                depth_skip = None
            depth -= 1
        elif depth_skip is None:
            if ch == "\\":
                m = re.match(r"\\(par|line|tab)\b ?|\\'([0-9a-fA-F]{2})|\\u(-?\d+)\??|\\[a-zA-Z]+-?\d* ?|\\(.)", data[i:])
                if m:
                    if m.group(1):
                        out.append("\t" if m.group(1) == "tab" else "\n")
                    elif m.group(2):
                        out.append(bytes([int(m.group(2), 16)]).decode("cp1252", errors="replace"))
                    elif m.group(3):
                        out.append(chr(int(m.group(3)) % 65536))
                    elif m.group(4):
                        out.append(m.group(4))
                    i += m.end()
                    continue
            elif ch not in "\r\n":
                out.append(ch)
        i += 1
    return _tidy("".join(out))


def _email(path: Path) -> tuple[str, str]:
    from email import policy
    from email.parser import BytesParser

    msg = BytesParser(policy=policy.default).parse(path.open("rb"))
    head = [f"{k}: {msg[k]}" for k in ("From", "To", "Cc", "Date", "Subject") if msg[k]]
    body = msg.get_body(preferencelist=("plain", "html"))
    text = ""
    if body is not None:
        text = body.get_content()
        if body.get_content_type() == "text/html":
            text = _html_text(text)
    attachments = [f"- {a.get_filename()} ({a.get_content_type()})" for a in msg.iter_attachments()]
    parts = ["\n".join(head), text.strip()]
    if attachments:
        parts.append("Attachments:\n" + "\n".join(attachments))
    return _tidy("\n\n".join(parts)), "email"


def _archive(path: Path) -> tuple[str, str]:
    """What is inside, never unpacked: names, sizes, and the total."""
    import tarfile

    entries: list[tuple[str, int]] = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            entries = [(i.filename, i.file_size) for i in z.infolist()]
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            entries = [(m.name + ("/" if m.isdir() else ""), m.size) for m in t.getmembers()]
    else:
        raise DocumentError(f"{path.name} is not a ZIP or TAR archive this can list")
    total = sum(size for _, size in entries)
    lines = [f"{size:>12,}  {name}" for name, size in entries[:MAX_LISTED]]
    if len(entries) > MAX_LISTED:
        lines.append(f"... and {len(entries) - MAX_LISTED} more")
    lines.append(f"{len(entries)} entries, {total:,} bytes unpacked. It was only listed, not extracted.")
    return "\n".join(lines) + "\n", "archive"


def _sqlite(path: Path) -> tuple[str, str]:
    """Tables, their columns and row counts; opened read-only, nothing is changed."""
    import sqlite3

    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as e:
        raise DocumentError(f"{path.name} could not be opened as a SQLite database ({e})") from e
    try:
        rows = con.execute("SELECT type, name, sql FROM sqlite_master WHERE type IN ('table','view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()
        out = []
        for kind, name, sql in rows:
            count = con.execute(f'SELECT COUNT(*) FROM "{name.replace(chr(34), chr(34) * 2)}"').fetchone()[0]
            out.append(f"## {kind} {name} ({count:,} rows)\n{sql};")
    except sqlite3.DatabaseError as e:
        raise DocumentError(f"{path.name} is not a SQLite database ({e})") from e
    finally:
        con.close()
    text = "\n\n".join(out) or "(no tables)"
    return text + "\n\nTo see rows, query it (e.g. with python -c \"import sqlite3 ...\"); this view is read-only.\n", \
        "SQLite database"
