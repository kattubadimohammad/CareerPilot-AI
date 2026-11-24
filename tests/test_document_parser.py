"""Tests for Document Parser."""
import io
import pytest
import docx
try:
    import pymupdf as fitz
except ImportError:
    import fitz

from src.document_parser import DocumentParser, DocumentValidationError


def create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_sample_docx(text: str) -> bytes:
    doc = docx.Document()
    for line in text.split("\n"):
        if line.strip():
            doc.add_paragraph(line)
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


def test_pdf_parsing():
    content = "Alice Johnson\nalice@example.com\nPython, Django, PostgreSQL"
    pdf_bytes = create_sample_pdf(content)
    extracted, file_type = DocumentParser.extract_text(pdf_bytes, "resume.pdf")

    assert file_type == "pdf"
    assert "Alice Johnson" in extracted
    assert "alice@example.com" in extracted
    assert "Python" in extracted


def test_docx_parsing():
    content = "Bob Smith\nbob@example.org\nDocker, Kubernetes, AWS"
    docx_bytes = create_sample_docx(content)
    extracted, file_type = DocumentParser.extract_text(docx_bytes, "resume.docx")

    assert file_type == "docx"
    assert "Bob Smith" in extracted
    assert "bob@example.org" in extracted
    assert "Kubernetes" in extracted


def test_validation_errors():
    with pytest.raises(DocumentValidationError, match="empty"):
        DocumentParser.validate_file(b"", "empty.pdf")

    with pytest.raises(DocumentValidationError, match="Unsupported file format"):
        DocumentParser.validate_file(b"%PDF-1.4 content", "resume.txt")

    with pytest.raises(DocumentValidationError, match="exceeds maximum"):
        DocumentParser.validate_file(b"%PDF-" + (b"x" * (11 * 1024 * 1024)), "large.pdf")


def test_magic_byte_validation():
    # File named .pdf but without PDF header
    with pytest.raises(DocumentValidationError, match="standard PDF header signature"):
        DocumentParser.validate_file(b"MZExecutableFileContent", "malicious.pdf")

    # File named .docx but without ZIP/PK header
    with pytest.raises(DocumentValidationError, match="standard ZIP/DOCX header signature"):
        DocumentParser.validate_file(b"TextContentNotZip", "malicious.docx")


def test_invalid_docx_structure_rejection():
    # Valid zip but not a docx (missing word/document.xml)
    zip_buffer = io.BytesIO()
    import zipfile
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("test.txt", "Some random text file")
    
    with pytest.raises(DocumentValidationError, match="not contain standard Word document structures"):
        DocumentParser.parse_docx(zip_buffer.getvalue())



def test_quick_contact_extraction():
    text = "Charlie Brown\nSoftware Engineer\ncharlie@peanuts.com\n(555) 123-4567\nSummary of experience..."
    contact = DocumentParser.quick_contact_extraction(text)
    assert contact["name"] == "Charlie Brown"
    assert contact["email"] == "charlie@peanuts.com"
    assert "123-4567" in contact["phone"]
