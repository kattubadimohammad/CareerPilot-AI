"""Document Parser module for PDF and DOCX resume extraction and validation."""
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pymupdf

import docx


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit
MAX_DECOMPRESSED_DOCX_BYTES = 25 * 1024 * 1024  # 25 MB max uncompressed
MAX_DECOMPRESSION_RATIO = 50.0  # Max compression ratio for zip-bomb defense
PDF_MAGIC_BYTES = b"%PDF-"
DOCX_MAGIC_BYTES = b"PK\x03\x04"


class DocumentValidationError(Exception):
    """Raised when document fails validation checks."""
    pass


class DocumentParser:
    """Handles extracting, cleaning, and validating text from PDF and DOCX files."""

    @staticmethod
    def validate_file(file_bytes: bytes, filename: str) -> str:
        """Validate file size, extension, and magic byte signatures. Returns normalized extension."""
        if file_bytes is None or len(file_bytes) == 0:
            raise DocumentValidationError("The uploaded file is empty (0 bytes). Please upload a valid document.")

        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            raise DocumentValidationError("File size exceeds maximum allowed limit (10MB).")

        clean_filename = Path(filename).name.replace("\x00", "")
        ext = Path(clean_filename).suffix.lower().lstrip(".")
        if not ext:
            raise DocumentValidationError("The file has no extension. Please upload a .pdf or .docx file.")

        if ext not in ["pdf", "docx"]:
            raise DocumentValidationError(f"Unsupported file format: .{ext}. Only PDF (.pdf) and Word (.docx) are supported.")

        # Magic byte signature checks
        if ext == "pdf":
            # Per ISO 32000-1 (PDF specification), %PDF- header may be preceded by UTF-8 BOM or header comment in first 1024 bytes
            if PDF_MAGIC_BYTES not in file_bytes[:1024]:
                raise DocumentValidationError("Invalid PDF: File does not contain standard PDF header signature (%PDF-).")
        elif ext == "docx":
            # Word DOCX is an OpenXML ZIP archive starting with PK signature
            if not file_bytes.startswith(b"PK"):
                raise DocumentValidationError("Invalid DOCX: File does not contain standard ZIP/DOCX header signature (PK).")

        return ext

    @staticmethod
    def parse_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            if doc.page_count == 0:
                raise DocumentValidationError("The PDF file contains 0 pages.")

            extracted_text = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text:
                    extracted_text.append(text)

            doc.close()
            full_text = "\n".join(extracted_text).strip()
            if not full_text:
                raise DocumentValidationError("Could not extract readable text from this PDF. It appears to be scanned or image-based. Please use a text-based PDF or DOCX file.")

            return DocumentParser.clean_text(full_text)
        except Exception as e:
            if isinstance(e, DocumentValidationError):
                raise
            raise DocumentValidationError(f"Failed to parse PDF: {str(e)}")

    @staticmethod
    def parse_docx(file_bytes: bytes) -> str:
        """Extract text from DOCX using python-docx with zip bomb protection."""
        try:
            # 1. Zip bomb and structure verification
            file_stream = io.BytesIO(file_bytes)
            with zipfile.ZipFile(file_stream, "r") as zf:
                total_uncompressed = sum(zinfo.file_size for zinfo in zf.infolist())
                if total_uncompressed > MAX_DECOMPRESSED_DOCX_BYTES:
                    raise DocumentValidationError(
                        f"DOCX uncompressed size ({total_uncompressed / (1024*1024):.1f}MB) exceeds safety limit (25MB)."
                    )

                compressed_size = len(file_bytes)
                if compressed_size > 0 and (total_uncompressed / compressed_size) > MAX_DECOMPRESSION_RATIO and total_uncompressed > 5 * 1024 * 1024:
                    raise DocumentValidationError("DOCX compression ratio indicates a potential zip bomb.")

                names = set(zf.namelist())
                if not any(n in names for n in ["word/document.xml", "[Content_Types].xml"]):
                    raise DocumentValidationError("Archive does not contain standard Word document structures.")

            # 2. Extract paragraphs and tables
            file_stream.seek(0)
            doc = docx.Document(file_stream)

            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            # Also check table cells
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        paragraphs.append(" | ".join(row_texts))

            full_text = "\n".join(paragraphs).strip()
            if not full_text:
                raise DocumentValidationError("Could not extract any readable text from DOCX file.")

            return DocumentParser.clean_text(full_text)
        except Exception as e:
            if isinstance(e, DocumentValidationError):
                raise
            raise DocumentValidationError(f"Failed to parse DOCX: {str(e)}")

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> Tuple[str, str]:
        """Validate and extract text from uploaded file. Returns (clean_text, file_type)."""
        file_type = cls.validate_file(file_bytes, filename)
        if file_type == "pdf":
            text = cls.parse_pdf(file_bytes)
        elif file_type == "docx":
            text = cls.parse_docx(file_bytes)
        else:
            raise DocumentValidationError(f"Unsupported format: {file_type}")

        return text, file_type

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize line breaks and excessive whitespace while preserving structure."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace 3 or more newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Replace non-breaking spaces or tabs with regular space
        text = text.replace("\u00a0", " ").replace("\t", " ")
        # Strip excessive spaces per line
        cleaned_lines = [re.sub(r"[ ]{2,}", " ", line).strip() for line in text.split("\n")]
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def quick_contact_extraction(text: str) -> Dict[str, Optional[str]]:
        """Extract email, phone, and plausible candidate name using regex heuristics."""
        email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        email = email_match.group(0) if email_match else None

        phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", text)
        phone = phone_match.group(0).strip() if phone_match else None

        # Look at the first non-empty lines for the candidate name
        name = None
        for line in text.split("\n")[:5]:
            line = line.strip()
            if line and len(line) <= 50 and not any(k in line.lower() for k in ["resume", "curriculum", "vitae", "summary", "profile", "contact", "@", "http", "phone"]):
                # Ensure it looks like a person's name (letters and spaces)
                if re.match(r"^[A-Z][A-Za-z\s.'-]{2,40}$", line):
                    name = line
                    break

        return {
            "name": name or "Candidate",
            "email": email,
            "phone": phone,
        }
