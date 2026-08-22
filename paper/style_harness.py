#!/usr/bin/env python3
"""Style harness: compare paper prose against a reference corpus of published ML papers.

Metrics computed for both the paper and the reference corpus, then compared:
  1. Sentence length (median, p90, % >40 words)
  2. Metaphor/colloquialism density (regex pattern list, max 1 per section)
  3. Terminology consistency (distinct names for same concept)
  4. Forward-reference density (Appendix/see §/Figure~\\ref per 1000 words)
  5. Number density (numbers per 100 words)
  6. Comma-splice index (sentences with 3+ comma-separated clauses)
  7. N-gram overlap (2-gram + 3-gram JSD divergence vs reference corpus)
  8. Vocabulary overlap (fraction of paper words in reference corpus)
  9. Passive voice ratio
  10. Abstract number count

Usage:
  python3 paper/style_harness.py                    # analyse default version
  python3 paper/style_harness.py --version v7       # analyse specific version
  python3 paper/style_harness.py --json             # machine-readable output
  python3 paper/style_harness.py --section 01_introduction  # single section

Exit codes:
  0 = all checks pass
  1 = one or more checks fail
"""

import re
import sys
import json
import math
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

# ─── stylometric metrics (Burrows Delta, TF-IDF cosine, function-word ratio) ─

FUNCTION_WORDS = set("""
the a an and or but in on at to for of with by from as is are was were be been being
have has had do does did will would can could should may might must this that these those
it its they them their there here where when how what which who whom whose
not no nor so than too very just only also even still yet already
we us our ours you your yours he him his she her hers
about above after again against all am any because before below between both during each
few further into more most other own same some such than through under until up down out
off over under again further then once
if though although unless since until while whereas whether
one two three first second third last next both either neither each every
said says say said make makes made go goes went get gets got take takes took come comes came
see sees saw know knows knew think thinks thought use uses used find finds found
""".split())


def cosine_similarity_vec(v1: dict, v2: dict) -> float:
    """Cosine similarity between two sparse vectors (dicts)."""
    if not v1 or not v2:
        return 0.0
    dot = sum(v1.get(k, 0) * v2.get(k, 0) for k in v1 if k in v2)
    mag1 = math.sqrt(sum(v * v for v in v1.values()))
    mag2 = math.sqrt(sum(v * v for v in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def relative_freq(tokens: list, word_list: set) -> dict:
    """Relative frequency of each function word in tokens."""
    freq = Counter(w for w in tokens if w in word_list)
    total = sum(freq.values()) or 1
    return {w: c / total for w, c in freq.items()}


def function_word_similarity(section_tokens: list, corpus_tokens: list) -> float:
    """Cosine similarity on function-word relative-frequency vectors.
    Measures syntactic style overlap independent of topic."""
    fw = FUNCTION_WORDS
    sec_vec = relative_freq(section_tokens, fw)
    corp_vec = relative_freq(corpus_tokens, fw)
    return cosine_similarity_vec(sec_vec, corp_vec)


def burrows_cosine_delta(section_tokens: list, corpus_doc_tokens: list) -> float:
    """Burrows' Cosine Delta: stylistic distance between a section and a corpus.

    Uses the N most frequent words (MFW) across the corpus. For each word,
    computes mean and std of relative frequency across corpus documents.
    Converts section and each corpus doc to z-score vectors, then returns
    the *minimum* cosine distance (most similar document). Lower = more similar.

    Works on short texts because it uses relative frequencies and cosine,
    both length-invariant.
    """
    N_MFW = 50  # number of most frequent words (tuned for small corpora)

    # Get word frequencies per corpus document
    doc_freqs = []
    for doc_tokens in corpus_doc_tokens:
        freq = Counter(doc_tokens)
        total = sum(freq.values()) or 1
        doc_freqs.append({w: c / total for w, c in freq.items()})

    # Find N most frequent words across entire corpus
    all_words = Counter()
    for doc_tokens in corpus_doc_tokens:
        all_words.update(doc_tokens)
    mfw = [w for w, _ in all_words.most_common(N_MFW)]

    # Compute mean and std per MFW across corpus documents
    means = {}
    stds = {}
    for w in mfw:
        vals = [df.get(w, 0) for df in doc_freqs]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        means[w] = mean
        stds[w] = math.sqrt(variance) if variance > 0 else 0.0001

    # Section z-scores
    sec_freq = Counter(section_tokens)
    sec_total = sum(sec_freq.values()) or 1
    sec_rel = {w: sec_freq.get(w, 0) / sec_total for w in mfw}
    sec_z = {w: (sec_rel[w] - means[w]) / stds[w] for w in mfw}

    # Compute cosine similarity to each corpus document, take the best (min distance)
    best_sim = 0.0
    for df in doc_freqs:
        doc_rel = {w: df.get(w, 0) for w in mfw}
        doc_z = {w: (doc_rel[w] - means[w]) / stds[w] for w in mfw}
        sim = cosine_similarity_vec(sec_z, doc_z)
        if sim > best_sim:
            best_sim = sim

    return 1.0 - best_sim  # cosine distance: lower = more similar


def sentence_length_similarity(section_text: str, corpus_sentence_lengths: list) -> float:
    """Overlap between sentence-length distributions of section and corpus.

    Builds histograms (1-60 words), normalizes, computes histogram intersection.
    Directly measures structural similarity: do your sentences have the same
    shape as the reference papers? 1.0 = identical distribution, 0.0 = no overlap.
    """
    sec_sentences = split_sentences(section_text)
    sec_lengths = [len(s.split()) for s in sec_sentences if len(s.split()) > 0]
    if not sec_lengths or not corpus_sentence_lengths:
        return 0.0

    # Build histograms (bins = 1 to 60 words, clip longer)
    max_bin = 60
    sec_hist = [0] * (max_bin + 1)
    for l in sec_lengths:
        sec_hist[min(l, max_bin)] += 1
    sec_total = sum(sec_hist) or 1
    sec_hist = [c / sec_total for c in sec_hist]

    corp_hist = [0] * (max_bin + 1)
    for l in corpus_sentence_lengths:
        corp_hist[min(l, max_bin)] += 1
    corp_total = sum(corp_hist) or 1
    corp_hist = [c / corp_total for c in corp_hist]

    # Histogram intersection (overlap coefficient)
    overlap = sum(min(a, b) for a, b in zip(sec_hist, corp_hist))
    return overlap


def tfidf_cosine_similarity(section_text: str, corpus_docs: list) -> float:
    """TF-IDF cosine similarity between a section and the best-matching corpus doc.

    Uses max similarity across docs rather than centroid, because with few
    reference papers the centroid dilutes the signal. Length-invariant.
    """
    # Build document frequency for IDF
    n_docs = len(corpus_docs)
    df = Counter()
    for doc in corpus_docs:
        seen = set(doc)
        for w in seen:
            df[w] += 1

    # IDF (smoothed)
    idf = {w: math.log((1 + n_docs) / (1 + d)) + 1 for w, d in df.items()}

    # TF-IDF vectors for each corpus document
    def tfidf_vector(tokens):
        tf = Counter(tokens)
        total = sum(tf.values()) or 1
        return {w: (c / total) * idf.get(w, 0) for w, c in tf.items()}

    doc_vectors = [tfidf_vector(doc) for doc in corpus_docs]

    # Section TF-IDF vector
    sec_tokens = [w for w in re.findall(r"[a-z]+", section_text.lower()) if len(w) > 1]
    sec_vec = tfidf_vector(sec_tokens)

    # Max similarity to any corpus document
    best = 0.0
    for dv in doc_vectors:
        sim = cosine_similarity_vec(sec_vec, dv)
        if sim > best:
            best = sim
    return best

REPO = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO / "paper" / "reference_corpus"
DEFAULT_VERSION = "v6"

# ─── colloquial/metaphor patterns ──────────────────────────────────────────

METAPHOR_PATTERNS = [
    r"\btaxi\b",
    r"\bsubway\b",
    r"\bfare\b",
    r"\bstation\b(?!s)",  # "station" but not "stations" in technical context
    r"\bblink\b",
    r"\btrail gone cold\b",
    r"\bemergency brake\b",
    r"before you (can )?finish",
    r"before you can blink",
    r"\bcatching\b",
    r"\bmetered\b",
    r"\bflat fare\b",
    r"\bper station\b",
    r"no matter how many",
    r"\bboard\b(?![a-z])",  # "board" but not "keyboard" etc
    r"\bthe gap (is |was )?(the )?difference\b",
    r"\bemergency\b(?!.*(?:service|room|shutdown))",
    r"\bknob\b",
    r"\bbrake\b",
]

METAPHOR_RE = re.compile("|".join(METAPHOR_PATTERNS), re.IGNORECASE)

# ─── terminology: concept → acceptable names ───────────────────────────────

TERMINOLOGY = {
    "block_diffusion": [
        r"\bblock-diffusion\b",
        r"\bblock diffusion\b(?!-)",  # not followed by hyphen
        r"\bDG\b(?![a-z])",
        r"\bdiffusion arm\b",
        r"\bblock-diffusion model\b",
        r"(?<!block-)\bdiffusion model\b",  # not preceded by "block-"
    ],
    "autoregressive": [
        r"\bautoregressive\b",
        r"\bauto-regressive\b",
        r"\bAR\b(?![a-z])",
        r"\bAR model\b",
        r"\bAR arm\b",
    ],
}

# ─── thresholds (derived from reference corpus analysis) ───────────────────

THRESHOLDS = {
    "sentence_median_max": 28,
    "sentence_median_min": 10,
    "sentence_p90_max": 45,
    "sentence_gt40_pct_max": 15.0,    # % of sentences >40 words
    "metaphor_max_per_section": 1,
    "terminology_names_max": 2,       # max distinct names for same concept
    "forward_ref_per_1k_max": 8.0,    # forward refs per 1000 words
    "number_density_max": 4.0,        # numbers per 100 words (main text)
    "comma_splice_pct_max": 10.0,     # % sentences with 3+ comma clauses
    "jsd_threshold": 0.35,            # JSD divergence threshold
    "vocab_overlap_min": 0.65,        # fraction of paper words in corpus
    "passive_pct_max": 25.0,          # % passive sentences
    "abstract_numbers_max": 3,
}

MAIN_SECTIONS = [
    "01_introduction", "02_cost_laws", "03_deployment",
    "04_deployment_losses", "05_survives", "06_related_work",
    "07_threats", "08_conclusion",
]

# ─── LaTeX cleaning ────────────────────────────────────────────────────────

def strip_latex(text: str) -> str:
    """Remove LaTeX commands, comments, math, and environments for prose analysis."""
    for env in ("equation", "align", "table", "tabular", "tabularx",
                "figure", "itemize", "enumerate", "abstract"):
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            " ",
            text,
            flags=re.DOTALL,
        )
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)
    text = re.sub(r"\$[^$]*\$", "X", text)
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", "", text)
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_latex_keep_structure(text: str) -> str:
    """Strip LaTeX but keep paragraph/sentence structure for prose analysis."""
    # Remove equation/align environments
    for env in ("equation", "align", "tabular", "tabularx"):
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            " ",
            text,
            flags=re.DOTALL,
        )
    # Remove table/figure environments but keep captions
    text = re.sub(r"\\begin\{table\*?\}", " [TABLE] ", text)
    text = re.sub(r"\\end\{table\*?\}", " [/TABLE] ", text)
    text = re.sub(r"\\begin\{figure\*?\}", " [FIGURE] ", text)
    text = re.sub(r"\\end\{figure\*?\}", " [/FIGURE] ", text)
    # Remove itemize/enumerate environments but keep items
    text = re.sub(r"\\begin\{(itemize|enumerate)\*?\}", "", text)
    text = re.sub(r"\\end\{(itemize|enumerate)\*?\}", "", text)
    text = re.sub(r"\\item\s*", "- ", text)
    # Remove footnotes
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)
    # Remove inline math $...$
    text = re.sub(r"\$[^$]*\$", "X", text)
    # Remove \command{...} but keep contents for some
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\textsc\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    # Remove other commands
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?\{[^}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    # Remove comments
    text = re.sub(r"(?<!\\)%.*", "", text)
    # Remove braces
    text = re.sub(r"[{}]", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_sections(version: str) -> dict:
    """Return {section_name: (cleaned_prose, raw_latex)} for all section files."""
    paper_dir = REPO / "paper" / version
    sections_dir = paper_dir / "sections"
    main_tex = paper_dir / "main.tex"
    sections = {}
    if sections_dir.exists():
        for f in sorted(sections_dir.iterdir()):
            if f.suffix == ".tex":
                raw = f.read_text()
                cleaned = strip_latex_keep_structure(raw)
                sections[f.stem] = cleaned
    if main_tex.exists():
        main_raw = main_tex.read_text()
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_raw, re.DOTALL)
        if m:
            sections["abstract"] = strip_latex_keep_structure(m.group(1))
    return sections


# ─── reference corpus loading ──────────────────────────────────────────────

def load_corpus() -> dict:
    """Load all reference corpus files. Returns {id: text}."""
    texts = {}
    manifest_path = CORPUS_DIR / "MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for paper in manifest["papers"]:
            p = CORPUS_DIR / paper["file"]
            if p.exists():
                texts[paper["id"]] = p.read_text()
    else:
        for p in sorted(CORPUS_DIR.glob("*.txt")):
            texts[p.stem] = p.read_text()
    return texts


def corpus_to_prose(text: str) -> str:
    """Strip metadata header from corpus file, return prose only."""
    lines = text.split("\n")
    # Skip header lines starting with #
    start = 0
    for i, line in enumerate(lines):
        if line.startswith("#"):
            start = i + 1
        elif line.strip() and start > 0:
            break
    return "\n".join(lines[start:])


# ─── text analysis utilities ───────────────────────────────────────────────

def split_sentences(text: str) -> list:
    """Split text into sentences."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if len(s.split()) > 3]


def count_words(text: str) -> int:
    return len(text.split())


def count_numbers(text: str) -> int:
    """Count numeric tokens in text."""
    return len(re.findall(r"\b\d[\d,.]*\b", text))


def find_metaphors(text: str) -> list:
    """Find colloquial/metaphorical phrases. Returns list of (match, position).
    Counts unique metaphorical sentences, not individual word matches."""
    results = []
    seen_sentences = set()
    for m in METAPHOR_RE.finditer(text):
        # Get surrounding context (sentence)
        start = max(0, text.rfind(".", 0, m.start()) + 1)
        end = text.find(".", m.end())
        if end == -1:
            end = len(text)
        sentence = text[start:end].strip()
        # Deduplicate by sentence — one metaphor per sentence counts as 1
        sent_key = sentence[:100]  # first 100 chars as key
        if sent_key in seen_sentences:
            continue
        seen_sentences.add(sent_key)
        results.append({
            "phrase": m.group(),
            "context": sentence[:200],
        })
    return results


def check_terminology(text: str) -> dict:
    """Check terminology consistency for each concept."""
    results = {}
    for concept, patterns in TERMINOLOGY.items():
        names_found = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Normalize the matched text, not the pattern
                for match in matches:
                    name = match.strip().lower()
                    # Group equivalent terms
                    if name in ("dg",):
                        name = "DG"
                    elif name in ("block-diffusion", "block diffusion", "block-diffusion model"):
                        name = "block-diffusion"
                    elif name in ("diffusion model", "diffusion arm"):
                        name = "diffusion model"
                    elif name in ("ar", "autoregressive", "auto-regressive"):
                        name = "AR/autoregressive"
                    elif name in ("ar model", "ar arm"):
                        name = "AR model/arm"
                    names_found.add(name)
        results[concept] = {
            "count": len(names_found),
            "names": sorted(names_found),
        }
    return results


def count_forward_refs(text: str) -> int:
    """Count forward references to appendices, sections, figures, tables."""
    patterns = [
        r"Appendix~?\\ref",
        r"Appendix~?\\ref\{",
        r"see [A-Z]ppendix",
        r"\\S\\ref",
        r"Section~?\\ref",
        r"Figure~?\\ref",
        r"Table~?\\ref",
        r"Eq\.~?\\ref",
        r"Eq\.~\\ref",
        r"Appendix",
    ]
    total = 0
    for p in patterns:
        total += len(re.findall(p, text))
    return total


def count_comma_splices(text: str) -> int:
    """Heuristic: count sentences with 3+ comma-separated independent clauses."""
    sentences = split_sentences(text)
    count = 0
    for s in sentences:
        # Count commas that separate clauses (rough heuristic)
        # Look for comma + space + lowercase word + verb pattern
        clauses = len(re.findall(r",\s+(?:and|but|or|so|yet|for|nor)\s+", s))
        # Also count comma-separated phrases that look like clauses
        comma_parts = s.split(",")
        if len(comma_parts) >= 4:  # 3+ commas = 4+ parts
            # Check if parts look like independent clauses (have verbs)
            verb_pattern = r"\b(?:is|are|was|were|be|been|has|have|had|do|does|did|will|would|can|could|should|may|might|must|pay|costs|grows|climbs|drops|wins|fails|holds|takes|makes|keeps|means|shows|provides|enables|requires|limits|constrains|narrows)\b"
            verb_parts = sum(1 for p in comma_parts if re.search(verb_pattern, p, re.IGNORECASE))
            if verb_parts >= 3:
                count += 1
                continue
        if clauses >= 2:
            count += 1
    return count


def check_passive(text: str) -> int:
    """Count sentences with passive voice constructions."""
    sentences = split_sentences(text)
    passive_re = re.compile(
        r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en|t)\b",
        re.IGNORECASE
    )
    count = 0
    for s in sentences:
        if passive_re.search(s):
            count += 1
    return count


# ─── n-gram analysis ───────────────────────────────────────────────────────

STOPWORDS = set("""
the a an and or but in on at to for of with by from as is are was were be been being
have has had do does did will would can could should may might must this that these those
it its it's they them their there here where when how what which who whom whose
not no nor so than too very just only also even still yet already
we us our ours you your yours he him his she her hers
i me my mine you your yours
s t re ve ll d m
about above after again against all am any because before below between both during each
few further into more most other own same some such than through under until up down out
off over under again further then once
""".split())


def tokenize(text: str) -> list:
    """Tokenize text into lowercase words, removing punctuation and stopwords."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def ngrams(tokens: list, n: int) -> Counter:
    """Generate n-gram frequency Counter from token list."""
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def jsd(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence between two distributions (as Counters)."""
    # Get all keys
    all_keys = set(p.keys()) | set(q.keys())
    if not all_keys:
        return 0.0
    # Normalize to probabilities
    p_sum = sum(p.values()) or 1
    q_sum = sum(q.values()) or 1
    # Compute JSD
    m = {}
    for k in all_keys:
        p_k = p.get(k, 0) / p_sum
        q_k = q.get(k, 0) / q_sum
        m[k] = (p_k + q_k) / 2
    # KL(P||M) and KL(Q||M)
    kl_pm = 0.0
    kl_qm = 0.0
    for k in all_keys:
        p_k = p.get(k, 0) / p_sum
        q_k = q.get(k, 0) / q_sum
        m_k = m[k]
        if p_k > 0 and m_k > 0:
            kl_pm += p_k * math.log2(p_k / m_k)
        if q_k > 0 and m_k > 0:
            kl_qm += q_k * math.log2(q_k / m_k)
    return (kl_pm + kl_qm) / 2


def vocab_overlap(paper_tokens: list, corpus_tokens: list) -> float:
    """Fraction of paper vocabulary that appears in corpus."""
    paper_vocab = set(paper_tokens)
    corpus_vocab = set(corpus_tokens)
    if not paper_vocab:
        return 0.0
    return len(paper_vocab & corpus_vocab) / len(paper_vocab)


# ─── main analysis ─────────────────────────────────────────────────────────

def analyze_section(name: str, text: str, corpus_text: str, corpus_tokens: list,
                    corpus_2grams: Counter, corpus_3grams: Counter,
                    corpus_doc_tokens: list = None,
                    corpus_raw_tokens: list = None,
                    corpus_sent_lengths: list = None) -> dict:
    """Analyze a single section against the reference corpus."""
    sentences = split_sentences(text)
    words = text.split()
    n_words = len(words) if words else 1

    # 1. Sentence length
    word_counts = [len(s.split()) for s in sentences] if sentences else [0]
    word_counts_sorted = sorted(word_counts)
    median_wc = word_counts_sorted[len(word_counts_sorted) // 2] if word_counts_sorted else 0
    p90_wc = word_counts_sorted[int(len(word_counts_sorted) * 0.9)] if word_counts_sorted else 0
    gt40 = sum(1 for c in word_counts if c > 40)
    gt40_pct = (gt40 / len(sentences) * 100) if sentences else 0

    # 2. Metaphors
    metaphors = find_metaphors(text)

    # 3. Terminology
    terms = check_terminology(text)

    # 4. Forward references
    fwd_refs = count_forward_refs(text)
    fwd_ref_per_1k = (fwd_refs / n_words * 1000) if n_words else 0

    # 5. Number density
    numbers = count_numbers(text)
    num_density = (numbers / n_words * 100) if n_words else 0

    # 6. Comma splices
    splices = count_comma_splices(text)
    splice_pct = (splices / len(sentences) * 100) if sentences else 0

    # 7. N-gram JSD
    tokens = tokenize(text)
    sec_2grams = ngrams(tokens, 2)
    sec_3grams = ngrams(tokens, 3)
    jsd_2 = jsd(sec_2grams, corpus_2grams) if sec_2grams else 1.0
    jsd_3 = jsd(sec_3grams, corpus_3grams) if sec_3grams else 1.0
    jsd_avg = (jsd_2 + jsd_3) / 2

    # 8. Vocabulary overlap
    vocab_ov = vocab_overlap(tokens, corpus_tokens)

    # 9. Passive voice
    passive_count = check_passive(text)
    passive_pct = (passive_count / len(sentences) * 100) if sentences else 0

    # 10. Stylometric metrics (new)
    all_tokens = re.findall(r"[a-z]+", text.lower())
    all_tokens = [w for w in all_tokens if len(w) > 1]
    # Function word similarity needs raw tokens (WITH function words)
    raw_corpus = corpus_raw_tokens if corpus_raw_tokens else corpus_tokens
    fw_sim = function_word_similarity(all_tokens, raw_corpus)
    cos_delta = burrows_cosine_delta(all_tokens, corpus_doc_tokens) if corpus_doc_tokens else 0.0
    tfidf_sim = tfidf_cosine_similarity(text, corpus_doc_tokens) if corpus_doc_tokens else 0.0
    sl_sim = sentence_length_similarity(text, corpus_sent_lengths) if corpus_sent_lengths else 0.0

    # Determine failures
    failures = []
    is_main = any(name.startswith(s) for s in MAIN_SECTIONS) or name == "abstract"
    is_abstract = name == "abstract"

    if is_main or is_abstract:
        if median_wc > THRESHOLDS["sentence_median_max"]:
            failures.append(f"sentence_median ({median_wc} > {THRESHOLDS['sentence_median_max']})")
        if median_wc < THRESHOLDS["sentence_median_min"]:
            failures.append(f"sentence_median ({median_wc} < {THRESHOLDS['sentence_median_min']})")
        if p90_wc > THRESHOLDS["sentence_p90_max"]:
            failures.append(f"sentence_p90 ({p90_wc} > {THRESHOLDS['sentence_p90_max']})")
        if gt40_pct > THRESHOLDS["sentence_gt40_pct_max"]:
            failures.append(f"sentence_gt40_pct ({gt40_pct:.1f}% > {THRESHOLDS['sentence_gt40_pct_max']}%)")

    if len(metaphors) > THRESHOLDS["metaphor_max_per_section"]:
        failures.append(f"metaphor_count ({len(metaphors)} > {THRESHOLDS['metaphor_max_per_section']})")

    for concept, info in terms.items():
        if info["count"] > THRESHOLDS["terminology_names_max"]:
            failures.append(f"terminology_{concept} ({info['count']} > {THRESHOLDS['terminology_names_max']})")

    if is_main or is_abstract:
        if fwd_ref_per_1k > THRESHOLDS["forward_ref_per_1k_max"]:
            failures.append(f"forward_ref_per_1k ({fwd_ref_per_1k:.1f} > {THRESHOLDS['forward_ref_per_1k_max']})")

    if is_main:
        if num_density > THRESHOLDS["number_density_max"]:
            failures.append(f"number_density ({num_density:.1f} > {THRESHOLDS['number_density_max']})")

    if splice_pct > THRESHOLDS["comma_splice_pct_max"]:
        failures.append(f"comma_splice_pct ({splice_pct:.1f}% > {THRESHOLDS['comma_splice_pct_max']}%)")

    # JSD is informational only — comparing 500-word sections to 78k-word corpus
    # always yields high divergence. Not a useful per-section failure criterion.
    # if jsd_avg > THRESHOLDS["jsd_threshold"]:
    #     failures.append(f"jsd_avg ({jsd_avg:.3f} > {THRESHOLDS['jsd_threshold']})")

    # Vocab overlap threshold is lower for appendices (more technical vocabulary)
    is_appendix = name.startswith("09") or name.startswith("1") and not name.startswith("19")
    vocab_threshold = 0.55 if is_appendix else THRESHOLDS["vocab_overlap_min"]
    if vocab_ov < vocab_threshold and n_words > 200:
        failures.append(f"vocab_overlap ({vocab_ov:.2f} < {vocab_threshold})")

    if is_main or is_abstract:
        if passive_pct > THRESHOLDS["passive_pct_max"]:
            failures.append(f"passive_pct ({passive_pct:.1f}% > {THRESHOLDS['passive_pct_max']}%)")

    if is_abstract:
        if numbers > THRESHOLDS["abstract_numbers_max"]:
            failures.append(f"abstract_numbers ({numbers} > {THRESHOLDS['abstract_numbers_max']})")

    # Long sentences for flagging
    long_sentences = []
    for i, (s, wc) in enumerate(zip(sentences, word_counts)):
        if wc > 40:
            long_sentences.append({"sentence": s[:300], "words": wc})

    return {
        "name": name,
        "n_words": n_words,
        "n_sentences": len(sentences),
        "metrics": {
            "sentence_median": median_wc,
            "sentence_p90": p90_wc,
            "sentence_gt40_pct": round(gt40_pct, 1),
            "metaphor_count": len(metaphors),
            "metaphors": metaphors,
            "terminology": terms,
            "forward_ref_per_1k": round(fwd_ref_per_1k, 1),
            "number_density": round(num_density, 1),
            "comma_splice_pct": round(splice_pct, 1),
            "jsd_2gram": round(jsd_2, 3),
            "jsd_3gram": round(jsd_3, 3),
            "jsd_avg": round(jsd_avg, 3),
            "vocab_overlap": round(vocab_ov, 3),
            "passive_pct": round(passive_pct, 1),
            "abstract_numbers": numbers if is_abstract else None,
            "burrows_cosine_delta": round(cos_delta, 4),
            "tfidf_cosine_sim": round(tfidf_sim, 4),
            "function_word_sim": round(fw_sim, 4),
            "sentence_length_sim": round(sl_sim, 4),
        },
        "long_sentences": long_sentences,
        "failures": failures,
        "pass": len(failures) == 0,
    }


def compute_corpus_baseline(corpus: dict) -> dict:
    """Compute baseline metrics from the reference corpus."""
    all_text = ""
    for text in corpus.values():
        all_text += corpus_to_prose(text) + "\n"

    sentences = split_sentences(all_text)
    word_counts = [len(s.split()) for s in sentences if len(s.split()) > 0]
    word_counts_sorted = sorted(word_counts)
    median_wc = word_counts_sorted[len(word_counts_sorted) // 2]
    p90_wc = word_counts_sorted[int(len(word_counts_sorted) * 0.9)]
    gt40_pct = sum(1 for c in word_counts if c > 40) / len(word_counts) * 100

    metaphors = find_metaphors(all_text)
    numbers = count_numbers(all_text)
    n_words = len(all_text.split())

    tokens = tokenize(all_text)
    corpus_2grams = ngrams(tokens, 2)
    corpus_3grams = ngrams(tokens, 3)

    # Per-document tokens for Burrows Delta and TF-IDF
    doc_tokens = []
    for text in corpus.values():
        doc_text = corpus_to_prose(text)
        doc_toks = re.findall(r"[a-z]+", doc_text.lower())
        doc_tokens.append([w for w in doc_toks if len(w) > 1])

    # Raw corpus tokens (WITH function words) for function-word similarity
    raw_tokens = re.findall(r"[a-z]+", all_text.lower())
    raw_tokens = [w for w in raw_tokens if len(w) > 1]

    # Corpus sentence lengths for sentence-length distribution similarity
    corpus_sent_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]

    return {
        "n_words": n_words,
        "n_sentences": len(sentences),
        "sentence_median": median_wc,
        "sentence_p90": p90_wc,
        "sentence_gt40_pct": round(gt40_pct, 1),
        "metaphor_count": len(metaphors),
        "number_density": round(numbers / n_words * 100, 1),
        "tokens": tokens,
        "2grams": corpus_2grams,
        "3grams": corpus_3grams,
        "doc_tokens": doc_tokens,
        "raw_tokens": raw_tokens,
        "corpus_sent_lengths": corpus_sent_lengths,
    }


def generate_report(results: list, baseline: dict, version: str) -> tuple:
    """Generate (json_report, markdown_report)."""
    # JSON
    json_report = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "corpus_baseline": {
            "n_words": baseline["n_words"],
            "n_sentences": baseline["n_sentences"],
            "sentence_median": baseline["sentence_median"],
            "sentence_p90": baseline["sentence_p90"],
            "sentence_gt40_pct": baseline["sentence_gt40_pct"],
            "metaphor_count": baseline["metaphor_count"],
            "number_density": baseline["number_density"],
        },
        "sections": results,
        "summary": {
            "total_sections": len(results),
            "passing": sum(1 for r in results if r["pass"]),
            "failing": sum(1 for r in results if not r["pass"]),
        },
    }

    # Markdown
    md = f"# Style Report — {version}\n\n"
    md += f"Generated: {json_report['timestamp']}\n\n"
    md += "## Reference Corpus Baseline\n\n"
    md += f"| Metric | Reference Corpus |\n|---|---|\n"
    md += f"| Words | {baseline['n_words']:,} |\n"
    md += f"| Sentences | {baseline['n_sentences']:,} |\n"
    md += f"| Sentence median (words) | {baseline['sentence_median']} |\n"
    md += f"| Sentence p90 (words) | {baseline['sentence_p90']} |\n"
    md += f"| Sentences >40 words (%) | {baseline['sentence_gt40_pct']}% |\n"
    md += f"| Metaphors found | {baseline['metaphor_count']} |\n"
    md += f"| Number density (per 100 words) | {baseline['number_density']} |\n\n"

    md += "## Section Results\n\n"
    md += f"| Section | Words | Sent | Med | p90 | >40% | Metaphor | JSD | Vocab | Delta | TF-IDF | FuncWord | SentLen | Status |\n"
    md += f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in results:
        m = r["metrics"]
        status = "PASS" if r["pass"] else "FAIL"
        md += f"| {r['name']} | {r['n_words']} | {r['n_sentences']} | {m['sentence_median']} | {m['sentence_p90']} | {m['sentence_gt40_pct']}% | {m['metaphor_count']} | {m['jsd_avg']} | {m['vocab_overlap']} | {m['burrows_cosine_delta']} | {m['tfidf_cosine_sim']} | {m['function_word_sim']} | {m['sentence_length_sim']} | {status} |\n"

    md += f"\n**Summary:** {json_report['summary']['passing']}/{json_report['summary']['total_sections']} sections pass\n\n"

    # Detailed failures
    failing = [r for r in results if not r["pass"]]
    if failing:
        md += "## Detailed Failures\n\n"
        for r in failing:
            md += f"### {r['name']}\n\n"
            md += f"**Failures:** {', '.join(r['failures'])}\n\n"

            m = r["metrics"]
            # Metaphors
            if m["metaphors"]:
                md += "**Metaphors found:**\n"
                for met in m["metaphors"]:
                    md += f"- \"{met['phrase']}\" in: \"{met['context']}\"\n"
                md += "\n"

            # Long sentences
            if r["long_sentences"]:
                md += "**Long sentences (>40 words):**\n"
                for ls in r["long_sentences"][:10]:
                    md += f"- ({ls['words']} words) \"{ls['sentence']}...\"\n"
                md += "\n"

            # Terminology
            for concept, info in m["terminology"].items():
                if info["count"] > THRESHOLDS["terminology_names_max"]:
                    md += f"**Terminology ({concept}):** {info['count']} distinct names: {', '.join(info['names'])}\n\n"

    # All metaphors across sections
    all_metaphors = []
    for r in results:
        for met in r["metrics"]["metaphors"]:
            all_metaphors.append((r["name"], met["phrase"], met["context"]))
    if all_metaphors:
        md += "## All Metaphors Found\n\n"
        for sec, phrase, ctx in all_metaphors:
            md += f"- **{sec}**: \"{phrase}\" — \"{ctx}\"\n"

    return json_report, md


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Style harness for paper prose")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Paper version dir (e.g., v6, v7)")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    parser.add_argument("--section", default=None, help="Analyse single section")
    parser.add_argument("--quiet", action="store_true", help="Only print failures")
    args = parser.parse_args()

    # Load corpus
    corpus = load_corpus()
    if not corpus:
        print("ERROR: No reference corpus found in paper/reference_corpus/", file=sys.stderr)
        sys.exit(2)

    baseline = compute_corpus_baseline(corpus)

    # Load paper sections
    sections = get_sections(args.version)
    if not sections:
        print(f"ERROR: No sections found in paper/{args.version}/", file=sys.stderr)
        sys.exit(2)

    # Filter to single section if requested
    if args.section:
        sections = {k: v for k, v in sections.items() if args.section in k}

    # Analyze each section
    results = []
    for name, text in sorted(sections.items()):
        r = analyze_section(
            name, text,
            corpus_to_prose(" ".join(corpus.values())),
            baseline["tokens"],
            baseline["2grams"],
            baseline["3grams"],
            baseline.get("doc_tokens"),
            baseline.get("raw_tokens"),
            baseline.get("corpus_sent_lengths"),
        )
        results.append(r)

    # Generate report
    json_report, md_report = generate_report(results, {
        "n_words": baseline["n_words"],
        "n_sentences": baseline["n_sentences"],
        "sentence_median": baseline["sentence_median"],
        "sentence_p90": baseline["sentence_p90"],
        "sentence_gt40_pct": baseline["sentence_gt40_pct"],
        "metaphor_count": baseline["metaphor_count"],
        "number_density": baseline["number_density"],
    }, args.version)

    # Output
    if args.json:
        print(json.dumps(json_report, indent=2))
    else:
        if not args.quiet:
            print(md_report)
        else:
            for r in results:
                if not r["pass"]:
                    print(f"FAIL: {r['name']} — {', '.join(r['failures'])}")

    # Save reports
    report_dir = REPO / "paper"
    json_path = report_dir / f"style_report_{args.version}.json"
    md_path = report_dir / f"style_report_{args.version}.md"
    json_path.write_text(json.dumps(json_report, indent=2))
    md_path.write_text(md_report)
    if not args.quiet:
        print(f"\nReports saved to {json_path} and {md_path}", file=sys.stderr)

    # Exit code
    n_failing = sum(1 for r in results if not r["pass"])
    sys.exit(1 if n_failing > 0 else 0)


if __name__ == "__main__":
    main()
