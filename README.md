# AI‑CoScientist – Golden Reference RAG Paper Collection

## Project Goal
Collect 100 recent papers (2020‑2025) from **Nature**, **Cell**, **Nature Medicine**, and **Science** that mention **"foundation model"**, and download their PDFs for use in the Golden Reference RAG system.

## Progress Summary

### Completed Steps
1. **Initial automation attempts** (`scripts/paper_collector.py`) failed due to SNU SSO redirects.
2. **Semi‑automated workflow** created:
   - `scripts/create_url_template.py` – template for manual URL entry.
   - `scripts/collect_paper_urls.py` – polls for the Nature search results page, saves authentication state, and collects URLs.
   - `scripts/download_papers_from_urls.py` – downloads PDFs, with a fallback to `requests` when PDFs open in‑browser.
3. **Authentication handling**:
   - Saved `auth_state.json` after manual login.
   - Scripts now load this state to avoid repeated logins.
4. **Pagination & URL collection**:
   - Robust page navigation and URL extraction.
   - Initially collected 20 URLs; later refined to search for "foundation model" papers.
5. **Search refinement**:
   - Updated `collect_paper_urls.py` to guide the user to a search URL that includes the query `"foundation model"` and the date range 2020‑2025.
6. **Download improvements**:
   - Added logic to detect in‑browser PDF view and download via session cookies.
   - Successfully downloaded several PDFs (e.g., *National food production…*).
7. **Artifacts updated**:
   - `implementation_plan.md`, `task.md`, `walkthrough.md` now reflect the evolving plan and status.

### Current Status
- **URL collection**: Script ready to collect papers containing the phrase *"foundation model"* from the target journals.
- **Downloads**: 20 URLs stored in `data/reference_papers/paper_urls.json`; several PDFs downloaded, others pending or lacking PDF links.

### Next Actions
1. **Run URL collection** (will open a browser for manual login/navigation):
   ```bash
   poetry run python scripts/collect_paper_urls.py
   ```
2. **Run PDF downloader** after URLs are collected:
   ```bash
   poetry run python scripts/download_papers_from_urls.py
   ```
3. **Verify each paper** contains the keyword "foundation model"; remove any that do not.

## How to Continue
- Adjust the search query in `collect_paper_urls.py` if you need to target other journals or keywords.
- The scripts will reuse the saved `auth_state.json` to keep you logged in.

---
*Generated on 2025‑11‑22 by Antigravity.*
