"""Paper editing and improvement services."""

# Lazy imports to avoid optional dependency issues
def __getattr__(name):
    """Lazy load modules to avoid importing optional dependencies."""
    if name == "PaperParser":
        from src.services.paper.parser import PaperParser
        return PaperParser
    elif name == "PaperAnalyzer":
        from src.services.paper.analyzer import PaperAnalyzer
        return PaperAnalyzer
    elif name == "PaperImprover":
        from src.services.paper.improver import PaperImprover
        return PaperImprover
    elif name == "PaperGenerator":
        from src.services.paper.generator import PaperGenerator
        return PaperGenerator
    elif name == "PaperExporter":
        from src.services.paper.exporter import PaperExporter
        return PaperExporter
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "PaperParser",
    "PaperAnalyzer",
    "PaperImprover",
    "PaperGenerator",
    "PaperExporter",
]
