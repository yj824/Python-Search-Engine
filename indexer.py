import os
import json
import re
import pickle
import hashlib
import warnings
from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
URL_SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.split("#", 1)[0]
    if url.endswith("/"):
        url = url[:-1]
    return url


def tokenize(text: str):
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def tokenize_and_stem(text: str, stemmer: PorterStemmer):
    if not text:
        return []
    tokens = tokenize(text)
    return [stemmer.stem(t) for t in tokens if t]


def tokenize_url_and_stem(url: str, stemmer: PorterStemmer):
    if not url:
        return []
    parts = URL_SPLIT_RE.split(url.lower())
    return [stemmer.stem(p) for p in parts if p]


def stable_hash_64(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False)


def compute_simhash(weighted_tf, bits=64):
    vector = [0] * bits

    for term, weight in weighted_tf.items():
        h = stable_hash_64(term)
        for i in range(bits):
            if (h >> i) & 1:
                vector[i] += weight
            else:
                vector[i] -= weight

    fingerprint = 0
    for i in range(bits):
        if vector[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def extract_document_features(html: str, url: str, stemmer: PorterStemmer):
    weighted_tf = defaultdict(int)

    if not html:
        return {
            "weighted_tf": weighted_tf,
            "doc_length": 0,
            "title_terms": [],
            "url_terms": [],
            "title_text": "",
            "content_hash": ""
        }

    soup = BeautifulSoup(html, "lxml")

    title_text = ""
    if soup.title and soup.title.get_text(strip=True):
        title_text = soup.title.get_text(" ", strip=True)
        for term in tokenize_and_stem(title_text, stemmer):
            weighted_tf[term] += 4

    headings = soup.find_all(["h1", "h2", "h3"])
    if headings:
        heading_text = " ".join(
            h.get_text(" ", strip=True) for h in headings if h.get_text(strip=True)
        )
        for term in tokenize_and_stem(heading_text, stemmer):
            weighted_tf[term] += 3

    bolds = soup.find_all(["strong", "b"])
    if bolds:
        bold_text = " ".join(
            b.get_text(" ", strip=True) for b in bolds if b.get_text(strip=True)
        )
        for term in tokenize_and_stem(bold_text, stemmer):
            weighted_tf[term] += 2

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    normal_text = soup.get_text(" ", strip=True)
    normal_terms = tokenize_and_stem(normal_text, stemmer)
    for term in normal_terms:
        weighted_tf[term] += 1

    url_terms = tokenize_url_and_stem(url, stemmer)
    for term in url_terms:
        weighted_tf[term] += 1

    content_hash = hashlib.md5(normal_text.lower().encode("utf-8")).hexdigest()

    return {
        "weighted_tf": weighted_tf,
        "doc_length": len(normal_terms),
        "title_terms": tokenize_and_stem(title_text, stemmer),
        "url_terms": url_terms,
        "title_text": title_text.lower(),
        "content_hash": content_hash
    }


def build_index(base_dir: str):
    stemmer = PorterStemmer()

    term_to_id = {}
    postings = defaultdict(list)
    doc_map = {}
    doc_stats = {}

    next_term_id = 0
    doc_id = 0
    processed = 0
    skipped = 0

    seen_urls = set()

    for root, _, files in os.walk(base_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue

            path = os.path.join(root, fn)

            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                url = normalize_url(obj.get("url", ""))
                html = obj.get("content", "")

                if not url or not html:
                    skipped += 1
                    continue

                if url in seen_urls:
                    skipped += 1
                    continue
                seen_urls.add(url)

                features = extract_document_features(html, url, stemmer)
                weighted_tf = features["weighted_tf"]

                if not weighted_tf:
                    skipped += 1
                    continue

                doc_id += 1
                doc_map[doc_id] = url

                doc_stats[doc_id] = {
                    "doc_length": features["doc_length"],
                    "title_terms": set(features["title_terms"]),
                    "url_terms": set(features["url_terms"]),
                    "title_text": features["title_text"],
                    "content_hash": features["content_hash"],
                    "simhash": compute_simhash(weighted_tf)
                }

                for term, tf in weighted_tf.items():
                    if term not in term_to_id:
                        term_to_id[term] = next_term_id
                        next_term_id += 1

                    term_id = term_to_id[term]
                    postings[term_id].append((doc_id, tf))

                processed += 1

                if processed % 200 == 0:
                    print(f"Processed {processed} docs... unique terms so far: {len(term_to_id)}")

            except Exception as e:
                skipped += 1
                print(f"Skipped {path}: {e}")

    return term_to_id, dict(postings), doc_map, doc_stats, processed, skipped


if __name__ == "__main__":
    BASE_DIR = "ANALYST"

    OUT_VOCAB = "vocab.pkl"
    OUT_POSTINGS = "postings.pkl"
    OUT_DOCMAP = "doc_map.pkl"
    OUT_DOCSTATS = "doc_stats.pkl"

    term_to_id, postings, doc_map, doc_stats, processed, skipped = build_index(BASE_DIR)

    save_pickle(term_to_id, OUT_VOCAB)
    save_pickle(postings, OUT_POSTINGS)
    save_pickle(doc_map, OUT_DOCMAP)
    save_pickle(doc_stats, OUT_DOCSTATS)

    print("=== DONE ===")
    print("Indexed documents:", processed)
    print("Skipped files:", skipped)
    print("Unique tokens:", len(term_to_id))
    print("Vocab file:", OUT_VOCAB)
    print("Postings file:", OUT_POSTINGS)
    print("Doc map file:", OUT_DOCMAP)
    print("Doc stats file:", OUT_DOCSTATS)