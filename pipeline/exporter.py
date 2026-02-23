"""
Export utilities for the Disclosure-to-Draft Pipeline.

Supports:
  • DOCX export via python-docx (with clickable footnote-style citations)
  • PDF  export via fpdf2 (with ✅ Human-Wall Approved stamp on cover page)
  • PDF  parsing of uploaded inventor files via pymupdf (fitz), with a
    plain-text fallback for memory-constrained environments (e.g. HF Spaces)

Both exports include a Provenance Appendix and an audit log summary.
"""

from __future__ import annotations

import io
import os
import re
from datetime import datetime, timezone
from typing import Optional

from pipeline.models import DraftSpecification, HumanWallChecklist


# ---------------------------------------------------------------------------
# PDF Input Parsing (for uploaded disclosures)
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """
    Extract plain text from an uploaded file.

    Supported formats: .pdf, .txt, .md, .docx (basic).

    Primary: pymupdf (fitz) for PDF parsing.
    Fallback: built-in text reader for .txt / .md, or basic byte scanning
              for PDFs when pymupdf is unavailable (safe for low-memory envs).
    """
    lower = file_path.lower()

    if lower.endswith(".txt") or lower.endswith(".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if lower.endswith(".pdf"):
        return _parse_pdf(file_path)

    if lower.endswith(".docx"):
        return _parse_docx_text(file_path)

    # Unknown format – try reading as UTF-8 text
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _parse_pdf(file_path: str) -> str:
    """Parse PDF with pymupdf; fall back to plain-text sniffer if unavailable."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n\n".join(pages)
    except ImportError:
        # pymupdf not installed – use lightweight fallback
        return _pdf_text_fallback(file_path)
    except Exception as e:
        print(f"[exporter] pymupdf error ({e}), using fallback")
        return _pdf_text_fallback(file_path)


def _pdf_text_fallback(file_path: str) -> str:
    """
    Bare-minimum PDF text extractor using built-in io/struct.
    Scans for BT...ET text blocks – handles simple text-layer PDFs only.
    This is the memory-safe fallback for Hugging Face Spaces.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        # Find all text between BT and ET markers
        text_blocks = re.findall(rb"BT\s+(.*?)\s+ET", data, re.DOTALL)
        lines = []
        for block in text_blocks:
            # Extract strings in parentheses: (text)
            strings = re.findall(rb"\(([^)]+)\)", block)
            for s in strings:
                try:
                    decoded = s.decode("latin-1").strip()
                    if decoded and len(decoded) > 2:
                        lines.append(decoded)
                except Exception:
                    pass
        return " ".join(lines) if lines else "[PDF text could not be extracted – please paste text directly]"
    except Exception:
        return "[PDF text extraction failed – please paste text directly]"


def _parse_docx_text(file_path: str) -> str:
    """Extract text from a .docx file using python-docx if available."""
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        # Fallback: treat as zip and read document.xml
        try:
            import zipfile
            with zipfile.ZipFile(file_path, "r") as z:
                with z.open("word/document.xml") as f:
                    xml = f.read().decode("utf-8", errors="ignore")
            # Strip XML tags
            text = re.sub(r"<[^>]+>", " ", xml)
            return " ".join(text.split())
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# DOCX Export
# ---------------------------------------------------------------------------

def export_to_docx(
    spec: DraftSpecification,
    checklist: HumanWallChecklist,
    output_path: str,
    session_id: Optional[str] = None,
) -> str:
    """
    Export the patent specification to a .docx file with:
      - Professional patent formatting
      - Footnote-style citations
      - Provenance appendix
      - Human-Wall approval record

    Args:
        spec:        DraftSpecification from Step 4.
        checklist:   HumanWallChecklist from Step 5.
        output_path: Full path for the output .docx file.
        session_id:  Pipeline session ID for audit (optional).

    Returns:
        The resolved output_path.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        raise RuntimeError(
            "python-docx is required for DOCX export. "
            "Install with: pip install python-docx"
        )

    doc = Document()

    # ── Cover page ──────────────────────────────────────────────────────
    _add_heading(doc, spec.title.upper(), level=0)
    _add_paragraph(doc, f"Patent Application Draft", bold=True, size=12)
    _add_paragraph(doc, f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if session_id:
        _add_paragraph(doc, f"Session ID: {session_id}", size=8)

    # Human-Wall approval stamp
    if checklist.export_allowed and checklist.user_approved_at:
        stamp_para = doc.add_paragraph()
        stamp_run = stamp_para.add_run(
            f"✅ HUMAN-WALL APPROVED — {checklist.user_approved_at.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        stamp_run.bold = True
        stamp_run.font.color.rgb = RGBColor(0x00, 0x80, 0x00)
        stamp_run.font.size = Pt(13)
    else:
        stamp_para = doc.add_paragraph()
        stamp_run = stamp_para.add_run("⚠️ DRAFT ONLY – NOT YET HUMAN-WALL APPROVED")
        stamp_run.bold = True
        stamp_run.font.color.rgb = RGBColor(0xFF, 0x66, 0x00)

    doc.add_page_break()

    # ── Specification sections ───────────────────────────────────────────
    _add_heading(doc, "ABSTRACT", level=1)
    doc.add_paragraph(spec.abstract)

    _add_heading(doc, "BACKGROUND", level=1)
    doc.add_paragraph(spec.background)

    _add_heading(doc, "SUMMARY OF THE INVENTION", level=1)
    doc.add_paragraph(spec.summary_of_invention)

    _add_heading(doc, "DETAILED DESCRIPTION", level=1)
    doc.add_paragraph(spec.detailed_description)

    if getattr(spec, "prior_art_navigation", None):
        _add_heading(doc, "PRIOR ART NAVIGATION STRATEGY", level=1)
        doc.add_paragraph(spec.prior_art_navigation)
        
    if getattr(spec, "competitive_positioning_map", None):
        _add_heading(doc, "COMPETITIVE POSITIONING MAP", level=1)
        doc.add_paragraph(spec.competitive_positioning_map)

    _add_heading(doc, "CLAIMS", level=1)
    for claim in spec.claims:
        dep = f" (depends on claim {claim.depends_on})" if claim.depends_on else ""
        p = doc.add_paragraph(style="List Number")
        p.add_run(f"Claim {claim.claim_number}{dep}: ").bold = True
        p.add_run(claim.text)
        # Strategy footnote
        footnote_para = doc.add_paragraph()
        fn_run = footnote_para.add_run(f"    [Strategy] {claim.strategy_reasoning}")
        fn_run.italic = True
        fn_run.font.size = Pt(9)
        fn_run.font.color.rgb = RGBColor(0x44, 0x44, 0x88)

    # ── Human-Wall Checklist summary ─────────────────────────────────────
    doc.add_page_break()
    _add_heading(doc, "HUMAN-WALL CHECKLIST", level=1)
    _add_paragraph(
        doc,
        f"Overall Confidence: {checklist.overall_confidence:.0%}  |  "
        f"Flags: {len(checklist.flags)} "
        f"(HIGH: {checklist.high_severity_count}, "
        f"MEDIUM: {checklist.medium_severity_count}, "
        f"LOW: {checklist.low_severity_count})",
        bold=True,
    )
    doc.add_paragraph(checklist.reviewer_notes)

    _add_heading(doc, "Flags for Attorney Attention", level=2)
    for flag in checklist.flags:
        p = doc.add_paragraph(style="List Bullet")
        sev_run = p.add_run(f"[{flag.severity.upper()}] [{flag.section}] ")
        sev_run.bold = True
        if flag.severity == "high":
            sev_run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
        elif flag.severity == "medium":
            sev_run.font.color.rgb = RGBColor(0xFF, 0x88, 0x00)
        p.add_run(f"{flag.issue} → {flag.suggestion}")

    # ── Provenance Appendix ──────────────────────────────────────────────
    doc.add_page_break()
    _add_heading(doc, "PROVENANCE APPENDIX", level=1)
    _add_paragraph(
        doc,
        "Every specification section traces back to the following disclosure sections:",
    )
    for spec_section, source_sections in spec.provenance_map.items():
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(f"{spec_section}: ").bold = True
        p.add_run(", ".join(source_sections))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# PDF Export
# ---------------------------------------------------------------------------

def export_to_pdf(
    spec: DraftSpecification,
    checklist: HumanWallChecklist,
    output_path: str,
    session_id: Optional[str] = None,
) -> str:
    """
    Export the patent specification to a .pdf file with:
      - ✅ Human-Wall Approved stamp on the cover page
      - All specification sections
      - Provenance appendix

    Args:
        spec:        DraftSpecification from Step 4.
        checklist:   HumanWallChecklist from Step 5.
        output_path: Full path for the output .pdf file.
        session_id:  Pipeline session ID for audit (optional).

    Returns:
        The resolved output_path.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError(
            "fpdf2 is required for PDF export. Install with: pip install fpdf2"
        )

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Cover page ──────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 20)
    pdf.multi_cell(0, 10, spec.title.upper(), align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Patent Application Draft", ln=True, align="C")
    pdf.cell(0, 8,
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
             ln=True, align="C")
    if session_id:
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 6, f"Session: {session_id}", ln=True, align="C")

    pdf.ln(10)

    # Human-Wall stamp
    if checklist.export_allowed and checklist.user_approved_at:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(0, 128, 0)  # Green
        pdf.multi_cell(
            0, 10,
            f"[APPROVED] HUMAN-WALL APPROVED\n"
            f"{checklist.user_approved_at.strftime('%Y-%m-%d %H:%M UTC')}",
            align="C"
        )
    else:
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_text_color(200, 80, 0)  # Orange
        pdf.multi_cell(0, 10, "[DRAFT ONLY] NOT YET HUMAN-WALL APPROVED", align="C")

    pdf.set_text_color(0, 0, 0)  # Reset to black
    pdf.add_page()

    # ── Specification sections ───────────────────────────────────────────
    _pdf_section(pdf, "ABSTRACT", spec.abstract)
    _pdf_section(pdf, "BACKGROUND", spec.background)
    _pdf_section(pdf, "SUMMARY OF THE INVENTION", spec.summary_of_invention)
    _pdf_section(pdf, "DETAILED DESCRIPTION", spec.detailed_description)

    if getattr(spec, "prior_art_navigation", None):
        _pdf_section(pdf, "PRIOR ART NAVIGATION STRATEGY", spec.prior_art_navigation)

    if getattr(spec, "competitive_positioning_map", None):
        _pdf_section(pdf, "COMPETITIVE POSITIONING MAP", spec.competitive_positioning_map)

    _pdf_heading(pdf, "CLAIMS")
    for claim in spec.claims:
        dep = f" (depends on claim {claim.depends_on})" if claim.depends_on else ""
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 6, f"Claim {claim.claim_number}{dep}:")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, claim.text)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(60, 60, 120)
        pdf.multi_cell(0, 5, f"  [Strategy] {claim.strategy_reasoning}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── Human-Wall Checklist ─────────────────────────────────────────────
    pdf.add_page()
    _pdf_heading(pdf, "HUMAN-WALL CHECKLIST")
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(
        0, 6,
        f"Overall Confidence: {checklist.overall_confidence:.0%}  |  "
        f"Flags: {len(checklist.flags)} "
        f"(HIGH: {checklist.high_severity_count}, "
        f"MEDIUM: {checklist.medium_severity_count}, "
        f"LOW: {checklist.low_severity_count})"
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, checklist.reviewer_notes)
    pdf.ln(4)

    for flag in checklist.flags:
        if flag.severity == "high":
            pdf.set_text_color(180, 0, 0)
        elif flag.severity == "medium":
            pdf.set_text_color(180, 100, 0)
        else:
            pdf.set_text_color(80, 80, 80)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, f"[{flag.severity.upper()}] [{flag.section}] {flag.issue}")
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, f"  Suggestion: {flag.suggestion}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # ── Provenance Appendix ──────────────────────────────────────────────
    pdf.add_page()
    _pdf_heading(pdf, "PROVENANCE APPENDIX")
    pdf.set_font("Helvetica", "", 9)
    for spec_section, sources in spec.provenance_map.items():
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(50, 6, spec_section + ":", ln=False)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 6, ", ".join(sources))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pdf.output(output_path)
    return output_path


# ---------------------------------------------------------------------------
# fpdf2 helpers
# ---------------------------------------------------------------------------

def _pdf_heading(pdf, text: str) -> None:
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(0, 0, 80)
    pdf.cell(0, 8, text, ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _pdf_section(pdf, heading: str, body: str) -> None:
    _pdf_heading(pdf, heading)
    pdf.set_font("Helvetica", "", 10)
    # fpdf2 multi_cell handles long text automatically
    safe_body = (body or "").encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 6, safe_body)
    pdf.ln(5)


# ---------------------------------------------------------------------------
# python-docx helpers
# ---------------------------------------------------------------------------

def _add_heading(doc, text: str, level: int) -> None:
    from docx.shared import Pt
    if level == 0:
        p = doc.add_heading(text, level=1)
        for run in p.runs:
            run.font.size = Pt(18)
    else:
        doc.add_heading(text, level=level)


def _add_paragraph(doc, text: str, bold: bool = False, size: int = 11) -> None:
    from docx.shared import Pt
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
