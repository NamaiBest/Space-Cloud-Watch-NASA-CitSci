#!/usr/bin/env python3
"""
Space Weather Gallery Scraper for NLC (Noctilucent Cloud) Images

Scrapes noctilucent cloud images from spaceweathergallery2.com and saves them
to a local folder with a CSV compatible with the NLC training pipeline.

All scraped images are NLC-positive (label=1). Type labels are left empty
because manual annotation would be required to determine type.

Usage:
    python scrape_spaceweather_gallery.py
    python scrape_spaceweather_gallery.py --pages 10 --output-dir spaceweather_gallery_images
    python scrape_spaceweather_gallery.py --start 0 --pages 20 --delay 1.0
"""

import os
import re
import time
import argparse
import hashlib
from pathlib import Path
from io import BytesIO

import requests
import pandas as pd
from PIL import Image

BASE_URL = "https://spaceweathergallery2.com"
DEFAULT_OUTPUT_DIR = "spaceweather_gallery_images"
DEFAULT_CSV_OUTPUT = "spaceweather_gallery_data.csv"
ITEMS_PER_PAGE = 25


def get_gallery_page(starting_point: int, session: requests.Session, timeout: int = 30):
    """Fetch a single gallery index page and return its HTML."""
    url = (
        f"{BASE_URL}/index.php"
        f"?title=noctilucent&title2=nlc&starting_point={starting_point}"
    )
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [!] Failed to fetch gallery page (offset={starting_point}): {e}")
        return None


def parse_gallery_page(html: str):
    """
    Extract upload entries directly from gallery index HTML.

    The gallery uses <a href='indiv_upload.php?upload_id=NNN'><img alt='...' src='...'></a>
    where alt contains: Title<br>Photographer<br>Date<br>Location
    and src is the thumbnail URL (submissions/pics/r/FILENAME_fpthumb.jpg).

    Returns list of dicts with: upload_id, title, photographer, date, location, thumb_url.
    """
    entries = []

    # Match: href='indiv_upload.php?upload_id=NNN' ... img src='...' alt='...'
    item_pattern = re.compile(
        r"href='indiv_upload\.php\?upload_id=(\d+)'[^>]*>"
        r"\s*<img\s+src='([^']+)'[^>]*alt='([^']*)'",
        re.DOTALL | re.IGNORECASE
    )

    for match in item_pattern.finditer(html):
        upload_id = match.group(1)
        thumb_src = match.group(2)
        alt_text = match.group(3)

        # alt text format: <font...>Title</font><br>Photographer<br>Date<br>Location
        # Replace <br> with newline before stripping all tags
        plain = re.sub(r'<br\s*/?>', '\n', alt_text, flags=re.IGNORECASE)
        plain = re.sub(r'<[^>]+>', '', plain)
        parts = [p.strip() for p in plain.split('\n') if p.strip()]

        title = parts[0] if len(parts) > 0 else ""
        photographer = parts[1] if len(parts) > 1 else ""
        date_str = parts[2] if len(parts) > 2 else ""
        location = parts[3] if len(parts) > 3 else ""

        # Ensure full thumb URL
        if thumb_src.startswith('/'):
            thumb_url = BASE_URL + thumb_src
        elif thumb_src.startswith('http'):
            thumb_url = thumb_src
        else:
            thumb_url = BASE_URL + '/' + thumb_src

        entries.append({
            "upload_id": upload_id,
            "title": title,
            "photographer": photographer,
            "date_str": date_str,
            "location": location,
            "thumb_url": thumb_url,
        })

    return entries


def build_image_urls(thumb_url: str):
    """
    Given a gallery thumbnail URL (…/pics/r/FILENAME_fpthumb.jpg),
    return candidate URLs in order of preference (largest → smallest).
    """
    candidates = []

    # Extract filename stem (without _fpthumb suffix)
    m = re.search(r'/pics/r/(.+?)(_fpthumb)?\.jpg', thumb_url, re.IGNORECASE)
    if m:
        stem = m.group(1)
        # /r/ without _fpthumb works (confirmed from gallery structure)
        candidates.append(f"{BASE_URL}/submissions/pics/r/{stem}.jpg")
        # Also try /m/ folder in case some uploads have it
        candidates.append(f"{BASE_URL}/submissions/pics/m/{stem}.jpg")

    # Always fall back to the thumbnail itself
    candidates.append(thumb_url)
    return candidates


def download_image(url: str, save_path: Path, session: requests.Session, timeout: int = 30) -> bool:
    """Download an image and save to disk. Returns True on success."""
    try:
        resp = session.get(url, timeout=timeout)
        resp.raise_for_status()

        # Verify it's a valid image
        img = Image.open(BytesIO(resp.content)).convert('RGB')

        # Require minimum size so thumbnails are large enough for training
        if img.width < 100 or img.height < 100:
            print(f"    [!] Image too small ({img.width}x{img.height}), skipping")
            return False

        img.save(str(save_path), 'JPEG', quality=90)
        return True

    except Exception as e:
        print(f"    [!] Download failed ({url}): {e}")
        return False


def scrape_gallery(
    output_dir: str,
    csv_output: str,
    start_offset: int = 0,
    num_pages: int = 10,
    delay: float = 0.75,
):
    """
    Main scraping function. Paginates through the NLC gallery, downloads images
    and saves a training-compatible CSV.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    })

    records = []
    seen_upload_ids = set()
    total_downloaded = 0

    print(f"Scraping {num_pages} pages from Space Weather Gallery (NLC filter)")
    print(f"Output folder : {output_path.resolve()}")
    print(f"Output CSV    : {csv_output}")
    print(f"Delay between requests: {delay}s\n")

    for page_num in range(num_pages):
        offset = start_offset + page_num * ITEMS_PER_PAGE
        print(f"--- Page {page_num + 1}/{num_pages} (starting_point={offset}) ---")

        html = get_gallery_page(offset, session)
        if not html:
            print("  [!] Could not fetch page, stopping.")
            break

        entries = parse_gallery_page(html)
        if not entries:
            print("  No entries found on this page — likely reached the end.")
            break

        print(f"  Found {len(entries)} entries")

        for entry in entries:
            upload_id = entry["upload_id"]
            if upload_id in seen_upload_ids:
                continue
            seen_upload_ids.add(upload_id)

            time.sleep(delay)

            print(f"  [{upload_id}] {entry['title'][:40]} | {entry['location'][:30]}")

            # Build candidate image URLs from the gallery thumbnail (no extra page fetch needed)
            candidates = build_image_urls(entry["thumb_url"])

            # Generate safe filename using upload_id
            m = re.search(r'/pics/r/(.+?)(_fpthumb)?\.jpg', entry["thumb_url"], re.IGNORECASE)
            stem = m.group(1) if m else upload_id
            safe_stem = re.sub(r'[^\w\-]', '_', stem)[:80]
            filename = f"{upload_id}_{safe_stem}.jpg"
            local_path = output_path / filename

            # Skip if already downloaded
            if local_path.exists():
                print(f"    [=] Already exists, skipping download")
            else:
                downloaded = False
                for img_url in candidates:
                    ok = download_image(img_url, local_path, session)
                    if ok:
                        print(f"    [+] Downloaded ({img_url.split('/')[-1]}) → {filename}")
                        total_downloaded += 1
                        downloaded = True
                        break

                if not downloaded:
                    print(f"    [!] All download attempts failed, skipping")
                    continue

            image_local_path = str(local_path.resolve())

            # Record metadata — compatible with parse_nlc_csv column names
            records.append({
                "observationid": f"swg_{upload_id}",
                "did you see nlc?": "Yes",
                "types of nlc": "",          # Unknown — type loss will be masked
                "latitude": "",
                "longitude": "",
                "observedDate": entry["date_str"],
                "locationName": entry["location"],
                "comments_observation": (
                    f"{entry['title']} by {entry['photographer']}"
                ),
                # Primary image column that parse_nlc_csv looks for
                "please upload an image of nlc (if available, optional)": image_local_path,
                "remote_image_url": entry["thumb_url"],
            })

        time.sleep(delay)

    # Save CSV
    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_output, index=False)
        print(f"\n✅ Saved {len(records)} records to '{csv_output}'")
        print(f"   Downloaded {total_downloaded} new images to '{output_dir}/'")
    else:
        print("\n⚠️  No records collected. Nothing saved.")

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Scrape NLC images from Space Weather Gallery for model training"
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to save downloaded images (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--csv-output", default=DEFAULT_CSV_OUTPUT,
        help=f"Output CSV filename (default: {DEFAULT_CSV_OUTPUT})"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Gallery starting_point offset to begin from (default: 0)"
    )
    parser.add_argument(
        "--pages", type=int, default=10,
        help="Number of gallery pages to scrape (25 images/page, default: 10)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.75,
        help="Seconds to wait between requests (default: 0.75)"
    )
    args = parser.parse_args()

    scrape_gallery(
        output_dir=args.output_dir,
        csv_output=args.csv_output,
        start_offset=args.start,
        num_pages=args.pages,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
