import math
import pickle
import re
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

LOW_VALUE_URL_HINTS = {
    "site-map", "sitemap", "tag", "tags", "archive", "category", "categories"
}

LOW_VALUE_TITLE_HINTS = {
    "site map", "sitemap", "tag", "tags", "archive", "category", "categories"
}

NEWSY_URL_HINTS = {
    "news", "quoted", "interview", "award", "awards", "featured", "honored",
    "wins", "win", "mentioned", "press", "media"
}

PROGRAM_URL_HINTS = {
    "grad", "undergrad", "program", "programs", "mswe", "mhcid", "phd-software-engineering",
    "ms-software-engineering", "bs-software-engineering"
}

FACULTY_URL_HINTS = {
    "faculty", "faculty-profiles", "faculty-profiles"
}


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def tokenize(text: str):
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())


def tokenize_and_stem(text: str, stemmer: PorterStemmer):
    if not text:
        return []
    tokens = tokenize(text)
    return [stemmer.stem(t) for t in tokens if t]


def hamming_distance(x: int, y: int) -> int:
    return bin(x ^ y).count("1")


def tf_weight(tf: int) -> float:
    if tf <= 0:
        return 0.0
    return 1.0 + math.log10(tf)


def compute_idf(num_docs: int, df: int) -> float:
    if df <= 0 or num_docs <= 0:
        return 0.0
    return math.log10(num_docs / df)


def build_postings_dict(postings_raw):
    postings = {}
    for term_id, plist in postings_raw.items():
        pdict = {doc_id: tf for doc_id, tf in plist}
        postings[term_id] = {
            "list": plist,
            "dict": pdict,
            "df": len(plist)
        }
    return postings


def precompute_doc_norms(postings, num_docs: int):
    doc_norm_sq = defaultdict(float)

    for term_id, entry in postings.items():
        df = entry["df"]
        idf = compute_idf(num_docs, df)
        if idf == 0:
            continue

        for doc_id, tf in entry["list"]:
            w_dt = tf_weight(tf) * idf
            doc_norm_sq[doc_id] += w_dt * w_dt

    doc_norms = {}
    for doc_id, sq in doc_norm_sq.items():
        doc_norms[doc_id] = math.sqrt(sq)

    return doc_norms


def intersect_sorted_doc_lists(doc_lists):
    if not doc_lists:
        return []

    result = set(doc_lists[0])
    for lst in doc_lists[1:]:
        result &= set(lst)

    return sorted(result)


class SearchEngine:
    def __init__(
        self,
        vocab_path="vocab.pkl",
        postings_path="postings.pkl",
        doc_map_path="doc_map.pkl",
        doc_stats_path="doc_stats.pkl"
    ):
        self.stemmer = PorterStemmer()

        self.term_to_id = load_pickle(vocab_path)
        self.postings = build_postings_dict(load_pickle(postings_path))
        self.doc_map = load_pickle(doc_map_path)
        self.doc_stats = load_pickle(doc_stats_path)

        self.num_docs = len(self.doc_map)
        self.doc_norms = precompute_doc_norms(self.postings, self.num_docs)

        print(f"Loaded {self.num_docs} docs, {len(self.term_to_id)} terms.")

    def _get_candidates(self, term_ids, require_all_terms=True):
        if require_all_terms:
            doc_lists = []
            for term_id in term_ids:
                doc_ids = [doc_id for doc_id, _ in self.postings[term_id]["list"]]
                doc_lists.append(doc_ids)

            candidates = intersect_sorted_doc_lists(doc_lists)
            if candidates:
                return candidates

        candidate_set = set()
        for term_id in term_ids:
            for doc_id, _ in self.postings[term_id]["list"]:
                candidate_set.add(doc_id)

        return sorted(candidate_set)

    def _score_tfidf_cosine(self, q_counter, candidate_docs):
        candidate_set = set(candidate_docs)
        scores = defaultdict(float)

        query_weights = {}
        for term, qtf in q_counter.items():
            term_id = self.term_to_id[term]
            df = self.postings[term_id]["df"]
            idf = compute_idf(self.num_docs, df)
            query_weights[term_id] = tf_weight(qtf) * idf

        query_norm = math.sqrt(sum(w * w for w in query_weights.values()))
        if query_norm == 0:
            return []

        for term_id, w_tq in query_weights.items():
            df = self.postings[term_id]["df"]
            idf = compute_idf(self.num_docs, df)

            for doc_id, tf in self.postings[term_id]["list"]:
                if doc_id not in candidate_set:
                    continue

                w_dt = tf_weight(tf) * idf
                scores[doc_id] += w_dt * w_tq

        ranked = []
        total_query_terms = len(q_counter)

        for doc_id, dot_product in scores.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)
            if doc_norm == 0:
                continue

            cosine = dot_product / (doc_norm * query_norm)

            matched_terms = 0
            for term in q_counter:
                term_id = self.term_to_id[term]
                if doc_id in self.postings[term_id]["dict"]:
                    matched_terms += 1

            coverage = matched_terms / total_query_terms if total_query_terms > 0 else 0.0
            final_score = cosine * (1.0 + 0.12 * coverage)

            ranked.append((doc_id, final_score))

        ranked.sort(key=lambda x: (-x[1], x[0]))
        return ranked

    def _is_program_like_query(self, raw_tokens):
        token_set = set(raw_tokens)
        degree_terms = {"master", "masters", "phd", "bs", "ms", "graduate", "undergraduate"}
        program_terms = {"software", "engineering", "interaction", "computer", "program", "programs"}
        return bool(token_set.intersection(degree_terms)) or (
            "software" in token_set and "engineering" in token_set
        ) or (
            "human" in token_set and "computer" in token_set and "interaction" in token_set
        )

    def _is_faculty_like_query(self, raw_tokens):
        token_set = set(raw_tokens)
        return "faculty" in token_set or ("informatics" in token_set and "faculty" in token_set)

    def _apply_boosts(self, query, q_counter, ranked_results):
        raw_tokens = tokenize(query)
        query_phrase = " ".join(raw_tokens).lower()
        query_terms = set(q_counter.keys())

        is_program_query = self._is_program_like_query(raw_tokens)
        is_faculty_query = self._is_faculty_like_query(raw_tokens)

        boosted = []

        for doc_id, score in ranked_results:
            stats = self.doc_stats[doc_id]
            url = self.doc_map[doc_id].lower()
            title_text = stats["title_text"]
            new_score = score

            title_overlap = len(query_terms.intersection(stats["title_terms"]))
            url_overlap = len(query_terms.intersection(stats["url_terms"]))

            if len(query_terms) > 0:
                new_score += 0.04 * title_overlap
                new_score += 0.03 * url_overlap

            if query_terms and query_terms.issubset(stats["title_terms"]):
                new_score += 0.12

            if query_phrase and query_phrase in title_text:
                new_score += 0.20

            if query_phrase and query_phrase.replace(" ", "-") in url:
                new_score += 0.18

            if is_program_query and any(hint in url for hint in PROGRAM_URL_HINTS):
                new_score += 0.16

            if is_faculty_query and any(hint in url for hint in FACULTY_URL_HINTS):
                new_score += 0.14

            # specific acronyms / short official program URLs
            if is_program_query and ("/grad/mswe" in url or "/grad/mhcid" in url):
                new_score += 0.18

            boosted.append((doc_id, new_score))

        boosted.sort(key=lambda x: (-x[1], x[0]))
        return boosted

    def _apply_penalties(self, query, ranked_results):
        raw_tokens = tokenize(query)
        is_program_query = self._is_program_like_query(raw_tokens)

        penalized = []

        for doc_id, score in ranked_results:
            url = self.doc_map[doc_id].lower()
            title_text = self.doc_stats[doc_id]["title_text"]

            new_score = score

            if any(hint in url for hint in LOW_VALUE_URL_HINTS):
                new_score -= 0.08

            if any(hint in title_text for hint in LOW_VALUE_TITLE_HINTS):
                new_score -= 0.06

            # 年份/新闻页轻微降权，尤其对 program-like query
            if any(hint in url for hint in NEWSY_URL_HINTS):
                new_score -= 0.04

            if is_program_query:
                if re.search(r"/20\d{2}/", url) or re.search(r"/20\d{2}-\d{2}/", url):
                    new_score -= 0.08
                if any(hint in url for hint in NEWSY_URL_HINTS):
                    new_score -= 0.08
                if "phd-student" in url or "phd-candidate" in url:
                    new_score -= 0.10

            penalized.append((doc_id, new_score))

        penalized.sort(key=lambda x: (-x[1], x[0]))
        return penalized

    def _remove_exact_duplicates(self, ranked_results):
        seen_hashes = set()
        filtered = []

        for doc_id, score in ranked_results:
            content_hash = self.doc_stats[doc_id]["content_hash"]
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            filtered.append((doc_id, score))

        return filtered

    def _remove_near_duplicates(self, ranked_results, threshold=3):
        filtered = []
        kept_ids = []

        for doc_id, score in ranked_results:
            current_hash = self.doc_stats[doc_id]["simhash"]
            duplicate = False

            for kept_id in kept_ids:
                kept_hash = self.doc_stats[kept_id]["simhash"]
                if hamming_distance(current_hash, kept_hash) <= threshold:
                    duplicate = True
                    break

            if not duplicate:
                filtered.append((doc_id, score))
                kept_ids.append(doc_id)

        return filtered

    def search(self, query: str, top_k=5, require_all_terms=True, simhash_threshold=3):
        query_terms = tokenize_and_stem(query, self.stemmer)
        if not query_terms:
            return []

        present_terms = [t for t in query_terms if t in self.term_to_id]
        if not present_terms:
            return []

        q_counter = Counter(present_terms)
        term_ids = [self.term_to_id[t] for t in q_counter.keys()]

        candidate_docs = self._get_candidates(term_ids, require_all_terms=require_all_terms)
        if not candidate_docs:
            return []

        ranked = self._score_tfidf_cosine(q_counter, candidate_docs)
        ranked = self._apply_boosts(query, q_counter, ranked)
        ranked = self._apply_penalties(query, ranked)
        ranked = self._remove_exact_duplicates(ranked)
        ranked = self._remove_near_duplicates(ranked, threshold=simhash_threshold)

        return ranked[:top_k]

    def search_urls(self, query: str, top_k=5, require_all_terms=True, simhash_threshold=3):
        results = self.search(
            query=query,
            top_k=top_k,
            require_all_terms=require_all_terms,
            simhash_threshold=simhash_threshold
        )
        return [(self.doc_map[doc_id], score) for doc_id, score in results]


if __name__ == "__main__":
    engine = SearchEngine()

    while True:
        query = input("\nEnter query (or type 'exit' to quit): ").strip()

        if query.lower() == "exit":
            break

        results = engine.search_urls(
            query=query,
            top_k=5,
            require_all_terms=True,
            simhash_threshold=3
        )

        if not results:
            print("No results.")
            continue

        for i, (url, score) in enumerate(results, 1):
            print(f"{i}. {url}  [score={score:.6f}]")