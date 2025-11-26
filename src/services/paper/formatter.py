from typing import Dict, Optional
import datetime

class PaperFormatter:
    """Formats the generated paper into various output formats."""

    def to_markdown(self, paper_data: Dict[str, str]) -> str:
        """Convert paper data to a formatted Markdown string."""
        title = paper_data.get("title", "Untitled Paper")
        authors = paper_data.get("authors", "AI-CoScientist")
        date = datetime.date.today().strftime("%B %d, %Y")
        
        md = f"""# {title}

**Authors:** {authors}
**Date:** {date}

---

## Abstract
{paper_data.get("abstract", "")}

## 1. Introduction
{paper_data.get("introduction", "")}

## 2. Methods
{paper_data.get("methods", "")}

## 3. Results
{paper_data.get("results", "")}

## 4. Discussion
{paper_data.get("discussion", "")}

## References
{paper_data.get("references", "References to be added.")}
"""
        return md

    def to_latex(self, paper_data: Dict[str, str]) -> str:
        """Convert paper data to a basic LaTeX document."""
        title = paper_data.get("title", "Untitled Paper")
        abstract = paper_data.get("abstract", "")
        
        latex = f"""\\documentclass{{article}}
\\usepackage{{authblk}}
\\usepackage{{geometry}}
\\geometry{{a4paper, margin=1in}}

\\title{{{title}}}
\\author{{AI-CoScientist}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\section{{Introduction}}
{paper_data.get("introduction", "")}

\\section{{Methods}}
{paper_data.get("methods", "")}

\\section{{Results}}
{paper_data.get("results", "")}

\\section{{Discussion}}
{paper_data.get("discussion", "")}

\\end{{document}}
"""
        return latex
