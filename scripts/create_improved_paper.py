#!/usr/bin/env python3
"""Create improved full paper by integrating enhancements with original."""

import re

# Read original paper
with open("/Users/jiookcha/Documents/git/AI-CoScientist/claudedocs/paper-before-extracted.txt") as f:
    original = f.read()

# Read improved excerpt
with open("/Users/jiookcha/Documents/git/AI-CoScientist/claudedocs/paper-revised-excerpt.txt") as f:
    improved_excerpt = f.read()

# Extract improved sections
def extract_section(text, start_marker, end_marker=None):
    """Extract section between markers."""
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None

    if end_marker:
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            return None
        return text[start_idx:end_idx]
    else:
        return text[start_idx:]

# Extract improved components
improved_title = extract_section(improved_excerpt, "", "\n\nJinwoo Lee").strip()
improved_abstract = extract_section(improved_excerpt, "Abstract\n\n", "\n\nIntroduction:")
improved_intro = extract_section(improved_excerpt, "Introduction: The Reproducibility Crisis", "\n\n[The rest of")
improved_discussion = extract_section(improved_excerpt, "Discussion: From Crisis to Paradigm Shift", "\n\n[Remainder")

# Find original sections to replace
original_title_end = original.find("\n\nJinwoo Lee")
original_abstract_start = original.find("Abstract\n\n") + len("Abstract\n\n")
original_abstract_end = original.find("Introduction\n\n")
original_intro_start = original.find("Introduction\n\n") + len("Introduction\n\n")
original_intro_end = original.find("\n\nBrief Overview of GRF")
original_discussion_start = original.find("Discussion\n\n") + len("Discussion\n\n")
original_discussion_end = original.find("\n\nLimitations")

# Build improved full paper
improved_full = ""

# Title and affiliations
improved_full += improved_title + "\n\n"
improved_full += original[original_title_end+2:original.find("Abstract\n\n")]

# Improved abstract
improved_full += "Abstract\n\n" + improved_abstract + "\n\n"

# Improved introduction
improved_full += "Introduction: The Reproducibility Crisis in Causal Machine Learning\n\n"
improved_full += improved_intro + "\n\n"

# Original methods (Brief Overview through Tutorial)
methods_start = original.find("Brief Overview of GRF")
methods_end = original.find("\n\nDiscussion\n\n")
improved_full += original[methods_start:methods_end] + "\n\n"

# Improved discussion
improved_full += "Discussion: From Crisis to Paradigm Shift\n\n"
improved_full += improved_discussion + "\n\n"

# Original limitations and beyond
limitations_start = original.find("Limitations")
improved_full += original[limitations_start:]

# Save improved full paper
output_path = "/Users/jiookcha/Desktop/paper-revised.txt"
with open(output_path, "w") as f:
    f.write(improved_full)

print(f"✅ Created improved full paper: {output_path}")
print(f"   Length: {len(improved_full)} characters")
print(f"   Original: {len(original)} characters")
print(f"   Change: {len(improved_full) - len(original):+d} characters")
