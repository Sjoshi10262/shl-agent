"""
SHL Product Catalog Scraper
Scrapes individual test solutions from https://www.shl.com/solutions/products/product-catalog/
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation & Preferences",
    "O": "Occupational Personality",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            print(f"  Attempt {attempt+1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_test_type(soup: BeautifulSoup) -> str:
    """Extract test type code from product page."""
    # Look for type indicators in the page
    text = soup.get_text()
    for code in TEST_TYPE_MAP:
        # Pattern: "Test Type: K" or similar
        if re.search(rf"\bTest Type[:\s]+{code}\b", text, re.IGNORECASE):
            return code
    # Look for badge/tag elements
    for tag in soup.find_all(["span", "div", "p"], class_=re.compile(r"type|badge|tag|label", re.I)):
        t = tag.get_text(strip=True).upper()
        if t in TEST_TYPE_MAP:
            return t
    return "K"  # Default to Knowledge & Skills


def parse_duration(soup: BeautifulSoup) -> str:
    text = soup.get_text()
    match = re.search(r"(\d+)\s*(?:–|-|to)\s*(\d+)\s*min", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{match.group(2)} minutes"
    match = re.search(r"(\d+)\s*minutes?", text, re.IGNORECASE)
    if match:
        return f"{match.group(1)} minutes"
    return ""


def parse_languages(soup: BeautifulSoup) -> list[str]:
    text = soup.get_text()
    match = re.search(r"Languages?[:\s]+([A-Za-z ,\n&]+?)(?:\n\n|\.|Test Type)", text)
    if match:
        langs = [l.strip() for l in re.split(r"[,\n&]+", match.group(1)) if l.strip()]
        return [l for l in langs if len(l) > 1 and len(l) < 50]
    return ["English"]


def parse_job_levels(text: str) -> list[str]:
    levels = []
    level_keywords = {
        "Entry": ["entry", "graduate", "junior", "associate"],
        "Mid-Professional": ["mid", "professional", "experienced"],
        "Manager": ["manager", "management", "supervisor"],
        "Senior Manager": ["senior manager", "director"],
        "Executive": ["executive", "c-suite", "vp", "vice president"],
        "General Population": ["general", "all levels"],
        "Frontline Manager": ["frontline manager", "team lead"],
    }
    text_lower = text.lower()
    for level, keywords in level_keywords.items():
        if any(kw in text_lower for kw in keywords):
            levels.append(level)
    return levels if levels else ["All Levels"]


def scrape_product_page(url: str) -> dict:
    """Scrape individual assessment product page."""
    soup = get_page(url)
    if not soup:
        return {}

    data = {"url": url}

    # Name
    h1 = soup.find("h1")
    data["name"] = h1.get_text(strip=True) if h1 else url.split("/")[-2].replace("-", " ").title()

    # Description - look for main content paragraphs
    description_parts = []
    content_area = soup.find("div", class_=re.compile(r"content|description|product|detail", re.I))
    if not content_area:
        content_area = soup.find("main") or soup.find("article") or soup
    
    for p in content_area.find_all("p", limit=5):
        text = p.get_text(strip=True)
        if len(text) > 50 and "cookie" not in text.lower():
            description_parts.append(text)
    data["description"] = " ".join(description_parts[:2]) if description_parts else data["name"]

    # Test type
    data["test_type"] = parse_test_type(soup)
    
    # Duration
    data["duration"] = parse_duration(soup)

    # Languages
    data["languages"] = parse_languages(soup)

    # Skills — look for bullet lists near skills/measures section
    skills = []
    full_text = soup.get_text()
    for section_header in ["measures", "skills", "competencies", "assesses"]:
        match = re.search(
            rf"{section_header}[:\s]+((?:[\w\s,/()-]+\n?)+)",
            full_text,
            re.IGNORECASE,
        )
        if match:
            raw = match.group(1)
            skills = [s.strip() for s in re.split(r"[,\n•·]+", raw) if s.strip() and len(s.strip()) > 2][:10]
            break
    data["skills"] = skills

    # Job levels
    data["job_levels"] = parse_job_levels(full_text)

    return data


def scrape_catalog() -> list[dict]:
    """Main scraper: get all individual test solutions from SHL catalog."""
    print(f"Fetching catalog: {CATALOG_URL}")
    
    assessments = []
    
    # The catalog uses pagination with start parameter
    page_start = 0
    page_size = 12
    
    seen_urls = set()
    
    while True:
        url = f"{CATALOG_URL}?type=1&start={page_start}"
        print(f"  Fetching page start={page_start}...")
        soup = get_page(url)
        if not soup:
            break

        # Find product cards/links
        product_links = []
        
        # SHL uses table rows or card elements for products
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "/products/" in href and href not in seen_urls:
                full_url = urljoin(BASE_URL, href)
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    product_links.append(full_url)

        if not product_links:
            # Try alternate selectors
            for row in soup.find_all("tr"):
                for link in row.find_all("a", href=True):
                    href = link["href"]
                    if "/products/" in href:
                        full_url = urljoin(BASE_URL, href)
                        if full_url not in seen_urls:
                            seen_urls.add(full_url)
                            product_links.append(full_url)

        print(f"  Found {len(product_links)} product links on this page")
        
        if not product_links:
            break

        for prod_url in product_links:
            print(f"  Scraping: {prod_url}")
            data = scrape_product_page(prod_url)
            if data.get("name"):
                assessments.append(data)
            time.sleep(0.5)

        # Check for next page
        next_btn = soup.find("a", string=re.compile(r"next|>", re.I))
        if not next_btn or page_start > 200:
            break
        page_start += page_size

    print(f"\nTotal assessments scraped: {len(assessments)}")
    return assessments


def save_catalog(assessments: list[dict], path: str = "data/shl_catalog.json"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(assessments)} assessments to {path}")


if __name__ == "__main__":
    assessments = scrape_catalog()
    if assessments:
        save_catalog(assessments)
    else:
        print("No assessments found — check site structure or use fallback catalog.")
