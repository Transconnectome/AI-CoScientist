"""File handling utilities for MCP server."""

from pathlib import Path
from typing import Optional
from docx import Document


def read_file(file_path: str) -> str:
    """Read text from file (.txt or .docx).

    Args:
        file_path: Path to the file

    Returns:
        Text content of the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check file extension
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    elif suffix == ".docx":
        doc = Document(str(path))
        # Extract text from all paragraphs
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return text

    else:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .txt, .docx")


def validate_file_path(file_path: str) -> Optional[str]:
    """Validate file path and return error message if invalid.

    Args:
        file_path: Path to validate

    Returns:
        Error message if invalid, None if valid
    """
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    suffix = path.suffix.lower()
    if suffix not in [".txt", ".docx"]:
        return f"Unsupported file format: {suffix}. Supported: .txt, .docx"

    return None
