"""
UM-Dearborn Fast Scraper using curl_cffi
Bypasses Cloudflare without a browser. Uses concurrent threads for speed.
"""

import os
import json
import time
import logging
import threading
from queue import Queue, Empty
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEED_URLS = [
    "https://umdearborn.edu",
    "https://umdearborn.edu/admissions",
    "https://umdearborn.edu/admissions-aid",
    "https://umdearborn.edu/admissions-aid/undergraduate",
    "https://umdearborn.edu/admissions-aid/undergraduate/ready-apply",
    "https://umdearborn.edu/admissions-aid/undergraduate/ready-apply/incoming-first-year-students",
    "https://umdearborn.edu/admissions-aid/undergraduate/ready-apply/transfer-students",
    "https://umdearborn.edu/admissions-aid/undergraduate/ready-apply/international-admissions",
    "https://umdearborn.edu/admissions-aid/undergraduate/ready-apply/high-school-counselors",
    "https://umdearborn.edu/admissions-aid/undergraduate/admitted-students",
    "https://umdearborn.edu/admissions-aid/undergraduate/visits-events",
    "https://umdearborn.edu/admissions-aid/undergraduate/explore-um-dearborn",
    "https://umdearborn.edu/admissions-aid/paying-college",
    "https://umdearborn.edu/admissions-aid/graduate-admissions",
    "https://umdearborn.edu/admissions-aid/graduate-admissions/how-apply",
    "https://umdearborn.edu/admissions-aid/graduate-admissions/graduate-tuition-and-funding",
    "https://umdearborn.edu/admissions-aid/apply",
    "https://umdearborn.edu/admissions-aid/campus-visits-and-events",
    "https://umdearborn.edu/academics",
    "https://umdearborn.edu/academics/programs",
    "https://umdearborn.edu/casl",
    "https://umdearborn.edu/cob",
    "https://umdearborn.edu/cecs",
    "https://umdearborn.edu/cehhs",
    "https://umdearborn.edu/student-life",
    "https://umdearborn.edu/campus-life",
    "https://umdearborn.edu/campus-life/housing",
    "https://umdearborn.edu/campus-life/dining",
    "https://umdearborn.edu/campus-life/student-organizations",
    "https://umdearborn.edu/campus-life/recreation-wellness",
    "https://umdearborn.edu/one-stop",
    "https://umdearborn.edu/one-stop/financial-aid",
    "https://umdearborn.edu/one-stop/financial-aid/types-aid/scholarships",
    "https://umdearborn.edu/one-stop/tuition-and-fees",
    "https://umdearborn.edu/one-stop/tuition-and-fees/cost-attendance",
    "https://umdearborn.edu/one-stop/registrar",
    "https://umdearborn.edu/research",
    "https://umdearborn.edu/about",
    "https://umdearborn.edu/about/history",
    "https://umdearborn.edu/about-um-dearborn/facts-and-figures",
    "https://umdearborn.edu/international-students",
    "https://umdearborn.edu/veterans-um-dearborn",
    "https://umdearborn.edu/library",
    "https://umdearborn.edu/news",
    "https://umdearborn.edu/alumni",
    "https://umdearborn.edu/giving",
    "https://umdearborn.edu/go-blue-guarantee",
    "https://umdearborn.edu/offices/career-services",
    "https://umdearborn.edu/offices/diversity-inclusion",
    "https://umdearborn.edu/offices/disability-services",
    "https://umdearborn.edu/offices/public-safety",
]

SKIP_PATTERNS = [
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".css", ".js", ".zip", ".doc", ".docx", ".ppt",
    "mailto:", "tel:", "javascript:", "#",
    "/calendar", "/feed", "/rss", "/print",
    "?page=", "?sort=", "&sort=",
]


def make_session() -> curl_requests.Session:
    s = curl_requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    return s


def extract_page(html: str, url: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript", "form"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    if not title and soup.find("h1"):
        title = soup.find("h1").get_text(strip=True)

    main = (soup.find("main") or soup.find(id="main-content") or
            soup.find(class_="main-content") or soup.find("article") or soup.body)

    paragraphs = []
    if main:
        # Extract tables as structured rows so short cells (e.g. "February 1")
        # keep their column header context instead of being filtered out.
        for table in main.find_all("table"):
            headers = [th.get_text(separator=" ", strip=True)
                       for th in table.find_all("th")]
            for row in table.find_all("tr"):
                cells = [td.get_text(separator=" ", strip=True)
                         for td in row.find_all("td")]
                if not cells:
                    continue
                if headers:
                    row_text = " | ".join(
                        f"{h}: {c}" for h, c in zip(headers, cells) if c
                    )
                else:
                    row_text = " | ".join(c for c in cells if c)
                if row_text:
                    paragraphs.append(row_text)
            table.decompose()  # avoid double-counting td/th below

        for elem in main.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6",
                                    "li", "dt", "dd"]):
            text = elem.get_text(separator=" ", strip=True)
            if len(text) > 20:
                paragraphs.append(text)
    content = "\n".join(paragraphs)

    # Collect same-domain links
    domain = urlparse(url).netloc
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].rstrip("/")
        if not href:
            continue
        full = urljoin(url, href)
        parsed = urlparse(full)
        if (parsed.netloc == domain and full.startswith("https://")
                and not any(p in full for p in SKIP_PATTERNS)):
            links.append(full)

    return {"url": url, "title": title, "content": content, "links": links}


def scrape_with_selenium(max_pages: int = 300, save_path: str = "data/scraped_pages.json",
                          extra_urls: List[str] = None, num_threads: int = 6) -> List[Dict]:
    """
    Scrape UM-Dearborn using curl_cffi (Cloudflare bypass) with concurrent threads.
    """
    logger.info(f"Starting fast scrape (max={max_pages}, threads={num_threads})")

    url_queue: Queue = Queue()
    results: List[Dict] = []
    visited: set = set()
    lock = threading.Lock()
    done_event = threading.Event()

    # Seed the queue
    seed = list(SEED_URLS)
    if extra_urls:
        seed.extend(extra_urls)
    for u in seed:
        url_queue.put(u.rstrip("/"))

    def worker():
        session = make_session()
        while not done_event.is_set():
            try:
                url = url_queue.get(timeout=3)
            except Empty:
                break

            url = url.rstrip("/")

            with lock:
                if url in visited or len(results) >= max_pages:
                    url_queue.task_done()
                    continue
                visited.add(url)
                current_count = len(results)

            logger.info(f"[{current_count+1}/{max_pages}] {url}")

            try:
                resp = session.get(url, impersonate="chrome120", timeout=12)
                if resp.status_code != 200:
                    logger.warning(f"  HTTP {resp.status_code} — skipping")
                    url_queue.task_done()
                    continue

                if "just a moment" in resp.text[:2000].lower():
                    logger.warning(f"  Cloudflare challenge — skipping")
                    url_queue.task_done()
                    continue

                data = extract_page(resp.text, url)

                if len(data["content"]) < 80:
                    logger.warning(f"  Too little content ({len(data['content'])} chars) — skipping")
                    url_queue.task_done()
                    continue

                with lock:
                    if len(results) < max_pages:
                        results.append(data)
                        logger.info(f"  OK {len(data['content'])} chars | {data['title'][:55]}")
                        if len(results) >= max_pages:
                            done_event.set()
                    # Enqueue discovered links
                    for link in data["links"]:
                        link = link.rstrip("/")
                        if link not in visited:
                            url_queue.put(link)

            except Exception as e:
                logger.warning(f"  Error: {e}")

            url_queue.task_done()
            time.sleep(0.3)  # polite delay per thread

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    logger.info(f"Scraped {len(results)} pages.")

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {save_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument("--output", default="data/scraped_pages.json")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    pages = scrape_with_selenium(max_pages=args.max_pages, save_path=args.output,
                                  num_threads=args.threads)
    print(f"\nDone! {len(pages)} pages scraped.")
