#!/usr/bin/env python3
"""Download required pretrained models for SOTA paper quality assessment."""

import sys
from pathlib import Path


def download_models():
    """Download SciBERT, RoBERTa, and DeBERTa models."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("❌ transformers not installed. Run: poetry install")
        sys.exit(1)

    models = [
        "allenai/scibert_scivocab_uncased",
        "roberta-base",
        "microsoft/deberta-xlarge-mnli"
    ]

    cache_dir = Path("./models/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DOWNLOADING PRETRAINED MODELS")
    print("=" * 80)

    for model_name in models:
        print(f"\n📥 Downloading {model_name}...")
        try:
            AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            print(f"✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModels cached in: {cache_dir.absolute()}")


def download_nlp_data():
    """Download spaCy and NLTK data."""
    print("\n" + "=" * 80)
    print("DOWNLOADING NLP DATA")
    print("=" * 80)

    # spaCy
    try:
        import spacy
        print("\n📥 Downloading spaCy model (en_core_web_sm)...")
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✅ spaCy model downloaded")
    except Exception as e:
        print(f"⚠️  spaCy download failed: {e}")
        print("   Run manually: python -m spacy download en_core_web_sm")

    # NLTK
    try:
        import nltk
        print("\n📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download failed: {e}")


if __name__ == "__main__":
    download_models()
    download_nlp_data()

    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Create validation dataset (50 papers with human scores)")
    print("2. Run: poetry run python scripts/validate_sota_methods.py")
