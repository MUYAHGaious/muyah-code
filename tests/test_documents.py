"""Read turns documents into text: Word, Excel, PowerPoint, OpenDocument, PDF, notebooks, e-books, RTF, emails,
archives and SQLite. Every fixture is built here, so no binary test files live in the repository."""

import json
import sqlite3
import tarfile
import zipfile
from email.message import EmailMessage

import pytest

from muyah_code.tools import documents
from muyah_code.tools.base import ToolError
from muyah_code.tools.fs import ReadTool

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_zip(path, files: dict):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return path


def docx(path, body: str):
    return make_zip(path, {"word/document.xml": f'<?xml version="1.0"?><w:document {W}><w:body>{body}</w:body></w:document>'})


def para(text, style=None, num=False):
    ppr = ""
    if style or num:
        ppr = "<w:pPr>" + (f'<w:pStyle w:val="{style}"/>' if style else "") + \
              ('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>' if num else "") + "</w:pPr>"
    return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"


def test_word_keeps_headings_lists_and_tables(tmp_path):
    table = ("<w:tbl>" + "".join(f"<w:tr><w:tc>{para(a)}</w:tc><w:tc>{para(b)}</w:tc></w:tr>"
                                 for a, b in [("Role", "Can do"), ("Buyer", "Browse, order"), ("Seller", "List items")])
             + "</w:tbl>")
    path = docx(tmp_path / "User_Roles_and_Flows.docx",
                para("User roles", "Heading1") + para("Everyone signs in first.") + para("Buyer", num=True)
                + para("Seller", num=True) + table)
    text, kind = documents.extract(path)
    assert kind == "Word document"
    assert "# User roles" in text and "Everyone signs in first." in text
    assert "- Buyer" in text and "- Seller" in text
    assert "| Role | Can do |" in text and "| Buyer | Browse, order |" in text


def test_excel_gives_every_sheet_as_rows(tmp_path):
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rel = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    path = make_zip(tmp_path / "prices.xlsx", {
        "xl/workbook.xml": f'<workbook {ns} {rel}><sheets><sheet name="Prices" sheetId="1" r:id="rId1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                                      '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        "xl/sharedStrings.xml": f'<sst {ns}><si><t>Item</t></si><si><t>Price</t></si><si><t>Kente, cloth</t></si></sst>',
        "xl/worksheets/sheet1.xml": f'<worksheet {ns}><sheetData>'
                                    '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>'
                                    '<row r="2"><c r="A2" t="s"><v>2</v></c><c r="B2"><v>45.5</v></c></row>'
                                    '<row r="3"><c r="B3"><f>SUM(B2:B2)</f></c></row>'
                                    '</sheetData></worksheet>'})
    text, kind = documents.extract(path)
    assert kind == "Excel workbook" and "## Sheet: Prices" in text
    assert "Item,Price" in text and '"Kente, cloth",45.5' in text and ",=SUM(B2:B2)" in text


def test_powerpoint_slide_by_slide(tmp_path):
    a = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'

    def slide(*paras):
        return f"<p:sld {a} xmlns:p='p'>" + "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paras) + "</p:sld>"
    path = make_zip(tmp_path / "pitch.pptx", {"ppt/slides/slide2.xml": slide("Plan", "Launch in May"),
                                               "ppt/slides/slide1.xml": slide("Market", "Growing 12%"),
                                               "ppt/slides/slide10.xml": slide("Thanks")})
    text, _ = documents.extract(path)
    assert text.index("Slide 1: Market") < text.index("Slide 2: Plan") < text.index("Slide 3: Thanks")
    assert "- Launch in May" in text


def test_opendocument_epub_rtf_and_email(tmp_path):
    odt = make_zip(tmp_path / "a.odt", {"content.xml": '<o xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
                                                       '<text:h text:outline-level="2">Scope</text:h>'
                                                       '<text:p>Only sellers list items.</text:p></o>'})
    assert "## Scope" in documents.extract(odt)[0] and "Only sellers" in documents.extract(odt)[0]

    epub = make_zip(tmp_path / "book.epub", {
        "META-INF/container.xml": '<container><rootfiles><rootfile full-path="OEBPS/c.opf"/></rootfiles></container>',
        "OEBPS/c.opf": '<package><manifest><item id="c1" href="one.xhtml"/></manifest><spine><itemref idref="c1"/></spine></package>',
        "OEBPS/one.xhtml": "<html><body><h1>Chapter One</h1><p>It began.</p><script>x()</script></body></html>"})
    text = documents.extract(epub)[0]
    assert "# Chapter One" in text and "It began." in text and "x()" not in text

    rtf = tmp_path / "note.rtf"
    rtf.write_text(r"{\rtf1\ansi{\fonttbl{\f0 Arial;}}\f0\fs24 Hello \b world\b0\par Caf\'e9 ok}", encoding="latin-1")
    text = documents.extract(rtf)[0]
    assert "Hello world" in text and "Café ok" in text and "Arial" not in text

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "a@x.com", "b@y.com", "Specs"
    msg.set_content("See the attached roles.")
    msg.add_attachment(b"PK", maintype="application", subtype="zip", filename="roles.zip")
    eml = tmp_path / "m.eml"
    eml.write_bytes(bytes(msg))
    text = documents.extract(eml)[0]
    assert "Subject: Specs" in text and "See the attached roles." in text and "roles.zip" in text


def test_archives_are_listed_never_unpacked(tmp_path):
    z = make_zip(tmp_path / "site.zip", {"index.html": "<h1>x</h1>", "css/app.css": "body{}"})
    text, kind = documents.extract(z)
    assert kind == "archive" and "index.html" in text and "css/app.css" in text and "not extracted" in text
    src = tmp_path / "f.txt"
    src.write_text("hi", encoding="utf-8")
    tgz = tmp_path / "b.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(src, arcname="pkg/f.txt")
    assert "pkg/f.txt" in documents.extract(tgz)[0]
    assert not (tmp_path / "pkg").exists() and not (tmp_path / "css").exists()


def test_sqlite_shows_the_schema_read_only(tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
    con.executemany("INSERT INTO products (name, price) VALUES (?, ?)", [("a", 1), ("b", 2)])
    con.commit()
    con.close()
    before = db.read_bytes()
    text, kind = documents.extract(db)
    assert kind == "SQLite database" and "table products (2 rows)" in text and "price REAL" in text
    assert db.read_bytes() == before


def test_notebook_cells_and_outputs(tmp_path):
    nb = {"metadata": {"kernelspec": {"language": "python"}}, "cells": [
        {"cell_type": "markdown", "source": ["# Analysis"]},
        {"cell_type": "code", "source": ["print(2 + 2)"], "outputs": [{"output_type": "stream", "text": ["4\n"]}]}]}
    path = tmp_path / "a.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    text, kind = documents.extract(path)
    assert "2 cells" in kind and "print(2 + 2)" in text and "Output:\n4" in text


def _pdf_with_pages(path, pages: list[str]):
    """A small valid PDF with one line of text per page (written by hand, byte offsets and all)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream.decode()}\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R "
                    "/Resources << /Font << /F1 3 0 R >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def test_pdf_pages_like_claude_code(tmp_path):
    pdf = _pdf_with_pages(tmp_path / "brief.pdf", [f"Page number {i}" for i in range(1, 26)])
    text, kind = documents.extract(pdf)
    assert kind == "PDF, 25 pages" and "Page number 1" in text and "Page number 20" in text
    assert "Page number 21" not in text and 'pages="21-25"' in text       # 20 at a time, and how to go on
    text, _ = documents.extract(pdf, "22-23")
    assert "--- page 22 ---" in text and "Page number 23" in text and "Page number 21" not in text
    with pytest.raises(documents.DocumentError, match="past the end"):
        documents.extract(pdf, "40")


def test_crafted_xml_and_oversized_parts_are_refused(tmp_path, monkeypatch):
    bomb = ('<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;&a;">]>'
            f'<w:document {W}><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>')
    path = make_zip(tmp_path / "evil.docx", {"word/document.xml": bomb})
    with pytest.raises(documents.DocumentError, match="entity"):
        documents.extract(path)
    monkeypatch.setattr(documents, "MAX_PART", 100)
    big = docx(tmp_path / "big.docx", para("x" * 500))
    with pytest.raises(documents.DocumentError, match="too big"):
        documents.extract(big)


def test_read_tool_shows_documents_as_numbered_text_and_old_doc_needs_libreoffice(ctx, monkeypatch):
    tmp_path = ctx.cwd
    path = docx(tmp_path / "spec.docx", para("Roles", "Heading1") + para("Buyers order."))
    res = ReadTool().run({"file_path": "spec.docx"}, ctx)
    assert "spec.docx (Word document, as text)" in res.content and "1\t# Roles" in res.content
    assert str(path) not in ctx.file_state          # a document is not something Edit can change

    (tmp_path / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0 old word file")
    monkeypatch.setattr(documents.shutil, "which", lambda name: None)
    monkeypatch.setattr(documents, "_windows_soffice", lambda: None)
    with pytest.raises(ToolError, match="LibreOffice"):
        ReadTool().run({"file_path": "old.doc"}, ctx)
