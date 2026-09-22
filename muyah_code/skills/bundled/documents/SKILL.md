---
name: documents
description: Use when asked to create or change a Word, Excel, PowerPoint, PDF or CSV file (a report, a spreadsheet, slides, a filled-in template) - writes a script with the right library so the result is a real document, then checks it by reading it back.
---
# Creating and changing documents

A `.docx`, `.xlsx`, `.pptx` or `.pdf` is not text: writing text into a file with that name makes a broken file.
Build it with a script, run it, then Read the result to check it (Read shows documents as text).

## 1. Pick the library

| Make | Library (pip name) | Notes |
|---|---|---|
| Word `.docx` | `python-docx` | headings, paragraphs, bullet lists, tables, images, page breaks |
| Excel `.xlsx` | `openpyxl` | several sheets, real formulas (`=SUM(B2:B9)`), number formats, column widths, bold headers, freeze panes |
| PowerPoint `.pptx` | `python-pptx` | title + bullet slides, images, simple tables, speaker notes |
| PDF | `reportlab` (layout) or `fpdf2` (simple) | or make a `.docx` and convert it with LibreOffice: `soffice --headless --convert-to pdf file.docx` |
| CSV / TSV | the standard `csv` module | `newline=""` and `encoding="utf-8-sig"` if it will open in Excel |
| Merge, split, rotate, fill PDFs | `pypdf` | already installed with MUYAH-CODE |

Check whether the library is there (`python -c "import docx"`). If not, install it with
`python -m pip install <name>`: this asks the user first, like any install. Prefer the project's own
environment when there is one (a venv, `uv`, `poetry`).

## 2. Write the script to a file, then run it

Put the script in the project (for example `scripts/make_report.py`) instead of a long `python -c`, so it can be
rerun and fixed. Keep content and layout apart: data at the top, building code below.

- **Word:** use built-in styles (`Heading 1`, `List Bullet`, `Table Grid`) instead of hand formatting, so the
  document gets a navigation pane and a table of contents works.
- **Excel:** write formulas, not computed numbers, when the user will change inputs. Set number formats
  (`#,##0.00`, `0%`, dates). Bold the header row, freeze it, size the columns.
- **Slides:** one idea per slide, 3-6 bullets, no paragraphs. Use the template's layouts when one is given.
- **PDF:** set margins and a font that has the characters you need (non-Latin text needs a TTF font registered).

## 3. Change an existing document

Open it with the library and change it in place, keeping its styles: `Document("in.docx")`,
`load_workbook("in.xlsx")`, `Presentation("in.pptx")`. Save to a new name first unless the user asked to
overwrite, and never delete the original: MUYAH-CODE does not delete files.

## 4. Check it

Read the file you made. Check that headings, tables, sheet names, formulas and slide titles are what was
asked. Say where the file is and what is in it. If LibreOffice is installed, converting to PDF is a good
extra check that the file opens.
