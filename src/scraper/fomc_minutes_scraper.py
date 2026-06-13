#!/usr/bin/env python3
"""
FedInsight - FOMC Minutes Scraper (improved version)

Features:
- Scrapes all historical FOMC minutes from federalreserve.gov (1994+)
- Extracts meeting_date, release_date, PDF URL
- Downloads PDFs with resume support (skips already downloaded)
- Saves rich metadata JSON
- Beautiful logging + progress bars
- Polite crawling (rate limiting)
- argparse for flexibility

Usage:
    python src/scraper/fomc_minutes_scraper.py
    python src/scraper/fomc_minutes_scraper.py --years 2020 2021 2022 --force
    python src/scraper/fomc_minutes_scraper.py --start-year 2015
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================
BASE_URL = "https://www.federalreserve.gov"
RAW_DIR = Path("data/raw/fomc_minutes")
PROCESSED_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DIR / "fomc_minutes_metadata.json"

# Polite crawling
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN_REQUESTS = 0.4  # seconds
SLEEP_BETWEEN_DOWNLOADS = 0.6

# User-Agent (be nice)
HEADERS = {
    "User-Agent": "FedInsight/1.0 (educational RAG project; contact: your@email.com)"
}


def setup_logging():
    """Configure nice logging with loguru"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )
    logger.add(
        PROCESSED_DIR / "scraper.log",
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
    )


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_meeting_date_from_filename(href: str) -> Optional[str]:
    """Extract YYYY-MM-DD from fomcminutes20190130.pdf"""
    match = re.search(r"fomcminutes(\d{4})(\d{2})(\d{2})\.pdf", href)
    if match:
        y, m, d = match.groups()
        return f"{y}-{m}-{d}"
    return None


def extract_release_date(text: str) -> Optional[str]:
    """Try to extract 'Released February 20, 2019' style string"""
    patterns = [
        r"Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})",
        r"Released\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def scrape_year(year: int) -> List[Dict]:
    """
    Scrape one historical year page and return list of meeting metadata dicts.
    """
    url = f"{BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
    logger.info(f"Scraping year {year} → {url}")

    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            logger.warning(f"Year {year} returned status {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")

        meetings: List[Dict] = []

        # Find all PDF links for minutes
        pdf_links = soup.find_all(
            "a", href=re.compile(r"/monetarypolicy/files/fomcminutes\d{8}\.pdf")
        )

        for link in pdf_links:
            href = link.get("href", "")
            if not href:
                continue

            full_url = BASE_URL + href if href.startswith("/") else href
            meeting_date = parse_meeting_date_from_filename(href)
            if not meeting_date:
                continue

            # Try to get release date from surrounding text
            release_date = None
            parent_text = ""
            if link.parent:
                parent_text = link.parent.get_text(separator=" ", strip=True)
            if not parent_text and link.find_parent("p"):
                parent_text = link.find_parent("p").get_text(separator=" ", strip=True)

            release_date = extract_release_date(parent_text)

            meetings.append(
                {
                    "meeting_date": meeting_date,
                    "release_date": release_date,
                    "pdf_url": full_url,
                    "source_year_page": year,
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                }
            )

        logger.success(f"Year {year}: found {len(meetings)} minutes PDFs")
        return meetings

    except requests.RequestException as e:
        logger.error(f"Network error for year {year}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error scraping year {year}: {e}")
        return []


def download_pdf(url: str, dest: Path) -> bool:
    """Download PDF with streaming and basic error handling"""
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def load_existing_metadata() -> List[Dict]:
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logger.warning("Could not load existing metadata, starting fresh")
    return []


def save_metadata(metadata: List[Dict]):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved → {METADATA_FILE} ({len(metadata)} records)")


def main():
    parser = argparse.ArgumentParser(description="FedInsight FOMC Minutes Scraper")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        help="Specific years to scrape (e.g. --years 2018 2019 2020)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1994,
        help="Start year for full historical scrape (default: 1994)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if PDF already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only collect metadata, do not download PDFs",
    )
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    logger.info("=== FedInsight FOMC Minutes Scraper ===")

    # Determine which years to process
    if args.years:
        years = sorted(set(args.years))
    else:
        current_year = datetime.now().year
        years = list(range(args.start_year, current_year + 1))

    logger.info(f"Will process years: {years[0]} → {years[-1]} ({len(years)} years)")

    # Load already known metadata (for future incremental logic)
    existing_metadata = load_existing_metadata()
    existing_urls = {m.get("pdf_url") for m in existing_metadata if m.get("pdf_url")}

    all_new_meetings: List[Dict] = []

    for year in tqdm(years, desc="Years", unit="year"):
        meetings = scrape_year(year)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        for meeting in meetings:
            pdf_url = meeting["pdf_url"]
            meeting_date = meeting["meeting_date"]
            filename = f"{meeting_date}_fomc_minutes.pdf"
            local_path = RAW_DIR / filename

            # Skip if we already have it and not forcing
            if local_path.exists() and not args.force:
                logger.debug(f"Already exists, skipping: {filename}")
                continue

            if pdf_url in existing_urls and not args.force:
                logger.debug(f"URL already in metadata, skipping download: {pdf_url}")
                continue

            if args.dry_run:
                logger.info(f"[DRY-RUN] Would download: {filename}")
                all_new_meetings.append(meeting)
                continue

            logger.info(f"Downloading {filename} ...")
            success = download_pdf(pdf_url, local_path)
            time.sleep(SLEEP_BETWEEN_DOWNLOADS)

            if success:
                meeting["local_path"] = str(local_path.resolve())
                meeting["downloaded_at"] = datetime.utcnow().isoformat() + "Z"
                all_new_meetings.append(meeting)
                logger.success(f"Saved → {local_path}")

    # Merge with existing metadata
    final_metadata = existing_metadata + all_new_meetings

    # Deduplicate by pdf_url
    seen = set()
    deduped = []
    for m in final_metadata:
        url = m.get("pdf_url")
        if url and url not in seen:
            seen.add(url)
            deduped.append(m)

    save_metadata(deduped)

    logger.success(
        f"Done! New downloads this run: {len(all_new_meetings)}. "
        f"Total in metadata: {len(deduped)}"
    )
    logger.info(f"PDFs location: {RAW_DIR.resolve()}")
    logger.info(f"Metadata: {METADATA_FILE.resolve()}")


if __name__ == "__main__":
    main()