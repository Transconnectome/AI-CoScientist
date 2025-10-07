#!/usr/bin/env python3
"""
Direct PDF Downloader for Top-5 Venues
Downloads PDFs directly from found URLs
"""

import requests
import os
import time

DOWNLOAD_DIR = "/Users/jiookcha/Documents/git/AI-CoScientist/neuroscience_ml_papers"

# Collected NeurIPS papers (brain/neuroscience/ML)
NEURIPS_PAPERS = [
    {
        'title': 'High-quality Video Reconstruction from Brain Activity',
        'url': 'https://papers.nips.cc/paper_files/paper/2023/file/4e5e0daf4b05d8bfc6377f33fd53a8f4-Paper-Conference.pdf',
        'venue': 'NeurIPS 2023'
    },
    {
        'title': 'Brain Dissection fMRI-trained Networks Reveal Spatial Selectivity',
        'url': 'https://papers.nips.cc/paper_files/paper/2023/file/90e06fe49254204248cb12562528b952-Paper-Conference.pdf',
        'venue': 'NeurIPS 2023'
    },
    {
        'title': 'Long-range Brain Graph Transformer',
        'url': 'https://papers.nips.cc/paper_files/paper/2024/file/2bd3ffba268a2699c212a233ed2907f1-Paper-Conference.pdf',
        'venue': 'NeurIPS 2024'
    },
    {
        'title': 'Can fMRI reveal the representation of syntactic structure',
        'url': 'https://papers.nips.cc/paper/2021/file/51a472c08e21aef54ed749806e3e6490-Paper.pdf',
        'venue': 'NeurIPS 2021'
    },
]

def download_pdf(url, filepath, title):
    """Download PDF with proper headers"""
