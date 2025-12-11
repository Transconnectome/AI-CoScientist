#!/usr/bin/env python3
"""Multi-agent peer review CLI.

This script now relies on ``src.services.review.adversarial`` so the same
logic can be reused inside the unified pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

import PyPDF2
from dotenv import load_dotenv

from src.services.review.adversarial import run_adversarial_review

load_dotenv()


def extract_pdf_text(pdf_path: Path) -> str:
    text: list[str] = []
    with pdf_path.open("rb") as handle:
        reader = PyPDF2.PdfReader(handle)
        for page in reader.pages:
            text.append(page.extract_text())
    return "\n".join(text)


def load_review_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent peer review utility")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to manuscript (PDF or text)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=Path,
        required=True,
        help="Path to reviewer prompt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for synthesized review output",
    )
    parser.add_argument(
        "--copy-to-repo",
        type=Path,
        help="Optional path to mirror the review file (e.g., Reviews repo)",
    )
    return parser.parse_args()


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_pdf_text(path)
    return path.read_text(encoding="utf-8")


def mirror_output(source_path: Path, destination_dir: Path) -> Optional[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / source_path.name
    target.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


async def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.prompt.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt}")

    document_text = read_document(args.input)
    prompt = load_review_prompt(args.prompt)

    print("=" * 80)
    print("MULTI-AGENT PEER REVIEW SYSTEM")
    print("=" * 80)
    print(f"📄 Manuscript: {args.input}")
    print(f"📝 Prompt: {args.prompt}")

    result = await run_adversarial_review(
        document_text=document_text,
        prompt=prompt,
        output_dir=args.output_dir,
        file_prefix="MULTI_AGENT_REVIEW",
    )

    print("=" * 80)
    print("PHASE 1: AGENT RESULTS")
    print("=" * 80)
    print(f"✅ Successful agents: {result.success_count}")
    print(f"❌ Failed agents: {result.failure_count}")

    if result.output_path:
        print(f"\n✅ Multi-agent review saved to: {result.output_path}")

    if args.copy_to_repo and result.output_path:
        mirrored = mirror_output(Path(result.output_path), args.copy_to_repo)
        print(f"✅ Copy saved to: {mirrored}")

    print("\nSummary JSON:")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
