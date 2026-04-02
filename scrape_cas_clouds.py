"""
Cloud Appreciation Society Cloud Library Scraper
-------------------------------------------------
Extracts cloud data (metadata + images) from:
  https://cloudappreciationsociety.org/cloud-library/

Output directory: "Cloud Appreciation Society data/"
  metadata.json               — all clouds consolidated
  {cloud_slug}/
      info.json               — name, description, altitude, precipitation, urls
      reference_images/       — curated reference photos
      gallery_images/         — recent community gallery photos

These images serve as hard-negative examples (tropospheric clouds)
for the NLC (Noctilucent Cloud) classifier.
"""

import json
import re
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://cloudappreciationsociety.org"
LIBRARY_BASE = f"{BASE_URL}/cloud-library/"
OUTPUT_DIR = Path("Cloud Appreciation Society data")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CitSci-NLC-Research/1.0)"}
TIMEOUT = 30
DELAY = 1.0          # polite crawl delay (seconds)
MAX_GALLERY = 20     # max gallery images to download per cloud type

# ── Seed cloud slugs ──────────────────────────────────────────────────────────
# Comprehensive starting set; crawler discovers more via page links.
SEED_SLUGS = [
    # 10 main genera
    "cumulus", "cumulonimbus", "stratus", "stratocumulus",
    "altostratus", "altocumulus", "nimbostratus",
    "cirrus", "cirrocumulus", "cirrostratus",
    # Special clouds
    "fog", "mist", "noctilucent", "contrail",
    "lenticularis", "mammatus", "mamma", "virga",
    "arcus", "roll-cloud", "pileus", "velum",
    "asperitas", "undulatus", "fractus", "fluctus",
    "fallstreak-hole", "cavum", "kelvin-helmholtz",
    # Optical phenomena (still clouds/atmosphere context)
    "corona", "glory", "cloudbow", "halo", "rainbow",
    "crepuscular-rays", "anticrepuscular-rays",
    "iridescence", "heiligenschein",
    # Others sometimes listed
    "nacreous", "polar-stratospheric",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_page(url: str) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except requests.RequestException as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None


def extract_bg_urls(soup_fragment) -> list[str]:
    """Extract image URLs from background-image CSS style attributes."""
    urls = []
    pattern = re.compile(r"url\(['\"]?(https?://[^'\")\s]+)['\"]?\)")
    for tag in soup_fragment.find_all(style=True):
        for match in pattern.finditer(tag["style"]):
            url = match.group(1)
            if "wp-content/uploads" in url:
                urls.append(url)
    return urls


def extract_section_images(soup: BeautifulSoup, section_title_keyword: str) -> list[str]:
    """Return image URLs from the section whose <h2> contains the keyword."""
    target_h2 = None
    for h2 in soup.find_all("h2"):
        if section_title_keyword.lower() in h2.get_text().lower():
            target_h2 = h2
            break
    if not target_h2:
        return []

    urls = []
    elem = target_h2.find_next_sibling()
    while elem and elem.name != "h2":
        urls.extend(extract_bg_urls(elem))
        # Also grab plain <img> uploads
        for img in elem.find_all("img", src=re.compile(r"uploads")):
            src = img.get("src", "")
            if src:
                urls.append(src)
        elem = elem.find_next_sibling()
    return urls


def extract_description(soup: BeautifulSoup) -> str:
    """Extract the 'About' text block."""
    about_h2 = None
    for h2 in soup.find_all("h2"):
        if "about" in h2.get_text().lower():
            about_h2 = h2
            break
    if not about_h2:
        return ""

    paragraphs = []
    elem = about_h2.find_next_sibling()
    while elem and elem.name not in ("h2", "h3"):
        if elem.name == "p":
            text = elem.get_text(separator=" ", strip=True)
            if text:
                paragraphs.append(text)
        elif elem.name in ("div", "section"):
            # look for <p> inside divs
            for p in elem.find_all("p"):
                text = p.get_text(separator=" ", strip=True)
                if text:
                    paragraphs.append(text)
        elem = elem.find_next_sibling()
    return " ".join(paragraphs)


def extract_altitude_precip(soup: BeautifulSoup) -> tuple[str, str]:
    altitude, precip = "", ""
    for h3 in soup.find_all("h3"):
        text = h3.get_text(strip=True)
        nxt = h3.find_next_sibling()
        if nxt is None:
            continue
        val = nxt.get_text(strip=True) if hasattr(nxt, "get_text") else ""
        if text == "Altitudes":
            altitude = val
        elif text == "Precipitation":
            precip = val
    return altitude, precip


def extract_linked_cloud_slugs(soup: BeautifulSoup) -> list[str]:
    """Find all /cloud-library/{slug}/ links on the page."""
    slugs = []
    pattern = re.compile(r"/cloud-library/([^/]+)/?$")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = pattern.search(href)
        if m:
            slug = m.group(1)
            if slug:
                slugs.append(slug)
    return list(set(slugs))


def download_image(url: str, dest_dir: Path) -> str | None:
    """Download image to dest_dir. Returns filename or None."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    ext = Path(urlparse(url).path).suffix or ".jpg"
    filename = f"{url_hash}{ext}"
    dest = dest_dir / filename
    if dest.exists():
        return filename
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet" not in content_type:
            return None
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except requests.RequestException as e:
        print(f"    [WARN] Image download failed {url}: {e}")
        return None


# ── Main crawl ────────────────────────────────────────────────────────────────

def scrape_cloud(slug: str, output_dir: Path) -> dict | None:
    """Scrape one cloud type page. Returns metadata dict or None if 404."""
    url = f"{LIBRARY_BASE}{slug}/"
    soup = get_page(url)
    if soup is None:
        return None

    # Confirm this is a valid cloud page (has an <h1>)
    h1 = soup.find("h1")
    if h1 is None:
        return None
    name = h1.get_text(strip=True)

    description = extract_description(soup)
    altitude, precipitation = extract_altitude_precip(soup)
    ref_image_urls = extract_section_images(soup, "reference images")
    gallery_image_urls = extract_section_images(soup, "gallery images")[:MAX_GALLERY]
    linked_slugs = extract_linked_cloud_slugs(soup)

    # Set up subdirectory
    cloud_dir = output_dir / slug
    ref_dir = cloud_dir / "reference_images"
    gal_dir = cloud_dir / "gallery_images"
    ref_dir.mkdir(parents=True, exist_ok=True)
    gal_dir.mkdir(parents=True, exist_ok=True)

    # Download reference images
    ref_files = []
    for img_url in ref_image_urls:
        fname = download_image(img_url, ref_dir)
        if fname:
            ref_files.append({"filename": fname, "url": img_url})
        time.sleep(0.2)

    # Download gallery images
    gal_files = []
    for img_url in gallery_image_urls:
        fname = download_image(img_url, gal_dir)
        if fname:
            gal_files.append({"filename": fname, "url": img_url})
        time.sleep(0.2)

    cloud_info = {
        "slug": slug,
        "name": name,
        "url": url,
        "description": description,
        "altitude": altitude,
        "precipitation": precipitation,
        "reference_images": ref_files,
        "gallery_images": gal_files,
        "linked_cloud_slugs": linked_slugs,
        "label": "non-nlc",        # hard-negative label for classifier
    }

    # Save per-cloud info.json
    with open(cloud_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(cloud_info, f, indent=2, ensure_ascii=False)

    return cloud_info


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    queue: list[str] = list(SEED_SLUGS)
    all_clouds: dict[str, dict] = {}

    print(f"Scraping Cloud Appreciation Society library → {OUTPUT_DIR}/")
    print(f"Starting with {len(queue)} seed slugs, will discover more via links.\n")

    with tqdm(total=len(queue), desc="Cloud types") as pbar:
        while queue:
            slug = queue.pop(0)
            if slug in visited:
                continue
            visited.add(slug)

            pbar.set_postfix_str(slug)
            info = scrape_cloud(slug, OUTPUT_DIR)
            time.sleep(DELAY)

            if info is None:
                pbar.update(1)
                continue

            all_clouds[slug] = info

            # Discover new cloud types from links
            for new_slug in info.get("linked_cloud_slugs", []):
                if new_slug not in visited and new_slug not in queue:
                    queue.append(new_slug)
                    pbar.total += 1
                    pbar.refresh()

            pbar.update(1)

    # Save consolidated metadata
    metadata = {
        "source": "Cloud Appreciation Society — https://cloudappreciationsociety.org/cloud-library/",
        "purpose": "Hard-negative examples for NLC (Noctilucent Cloud) binary classifier",
        "total_cloud_types": len(all_clouds),
        "clouds": all_clouds,
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'─'*60}")
    print(f"Done! Scraped {len(all_clouds)} cloud types.")
    ref_total = sum(len(v['reference_images']) for v in all_clouds.values())
    gal_total = sum(len(v['gallery_images']) for v in all_clouds.values())
    print(f"  Reference images downloaded : {ref_total}")
    print(f"  Gallery images downloaded   : {gal_total}")
    print(f"  Total images                : {ref_total + gal_total}")
    print(f"  Metadata saved to           : {meta_path}")
    print(f"  Output directory            : {OUTPUT_DIR.resolve()}")
    print(f"\nCloud types found:")
    for slug, info in sorted(all_clouds.items()):
        n_ref = len(info['reference_images'])
        n_gal = len(info['gallery_images'])
        print(f"  {info['name']:<30}  ref:{n_ref:>2}  gallery:{n_gal:>2}")


if __name__ == "__main__":
    main()
