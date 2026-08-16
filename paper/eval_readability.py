#!/usr/bin/env python3
"""Evaluate paper readability and expression density.

Two scripts in one file:
  - readability: Flesch Reading Ease, sentence length, passive voice, hedging
  - density: number density, bracket density, words-per-number

Usage:
  python3 paper/eval_readability.py            # both
  python3 paper/eval_readability.py --read      # readability only
  python3 paper/eval_readability.py --density   # density only
  python3 paper/eval_readability.py --json      # machine-readable

Exit codes:
  0 = all checks pass
  1 = one or more checks fail (see thresholds below)
"""

import re
import os
import sys
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper" / "v4"
SECTIONS_DIR = PAPER / "sections"
MAIN_TEX = PAPER / "main.tex"

# ─── thresholds ────────────────────────────────────────────────────────────

THRESHOLDS = {
    # readability
    "flesch_main_min": 18,       # main text sections (abstract can be denser)
    "flesch_main_max": 55,
    "flesch_appendix_min": 20,   # appendices can be denser
    "median_sentence_min": 10,   # words (short appendix sentences are OK)
    "median_sentence_max": 28,
    "sentences_gt40_max": 2,    # per section (excl equation/table envs)
    "sentences_gt60_max": 0,    # per section
    "passive_cluster_max": 3,   # consecutive passive sentences
    "hedge_per_sentence_max": 0.5,
    # density
    "number_density_main_max": 4.0,   # numbers per 100 words in main text
    "number_density_abs_max": 5.0,   # any single section
    "bracket_density_main_max": 1.5, # brackets per 100 words (related work has citation brackets)
    "words_per_number_min": 25,       # at least 25 words between numbers
    "abstract_numbers_max": 3,       # inline numbers in abstract prose
}

MAIN_SECTIONS = [
    "01_introduction", "02_cost_laws", "03_deployment",
    "04_deployment_losses", "05_survives", "06_related_work",
    "07_threats", "08_conclusion",
]

# ─── LaTeX cleaning ────────────────────────────────────────────────────────

def strip_latex(text: str) -> str:
    """Remove LaTeX commands, comments, math, and environments for prose analysis."""
    # Remove equation/align/table/tabular environments entirely
    for env in ("equation", "align", "table", "tabular", "tabularx", "figure", "itemize", "enumerate"):
        text = re.sub(
            rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}",
            " ",
            text,
            flags=re.DOTALL,
        )
    # Remove footnotes
    text = re.sub(r"\\footnote\{[^}]*\}", "", text)
    # Remove inline math $...$
    text = re.sub(r"\$[^$]*\$", "X", text)
    # Remove \texttt{...}, \emph{...}, \textbf{...}, etc.
    text = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", "", text)
    # Remove comments
    text = re.sub(r"%.*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sections() -> dict:
    """Return {section_name: cleaned_text} for all section files."""
    sections = {}
    for f in sorted(SECTIONS_DIR.iterdir()):
        if f.suffix == ".tex":
            raw = f.read_text()
            sections[f.stem] = strip_latex(raw)
    # Add abstract
    if MAIN_TEX.exists():
        main_raw = MAIN_TEX.read_text()
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_raw, re.DOTALL)
        if m:
            sections["abstract"] = strip_latex(m.group(1))
    return sections


def split_sentences(text: str) -> list:
    """Split text into sentences, filtering very short fragments."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.split()) > 3]


# ─── readability metrics ───────────────────────────────────────────────────

def flesch_reading_ease(text: str) -> float:
    """Compute Flesch Reading Ease without external dependencies."""
    sentences = split_sentences(text)
    words = text.split()
    if not sentences or not words:
        return 0.0
    # Count syllables (approximate)
    syllables = 0
    for word in words:
        clean = re.sub(r"[^a-zA-Z]", "", word).lower()
        if not clean:
            continue
        count = len(re.findall(r"[aeiouy]+", clean))
        syllables += max(1, count)
    asl = len(words) / len(sentences)  # avg sentence length
    asw = syllables / len(words)        # avg syllables per word
    return 206.835 - 1.015 * asl - 84.6 * asw


def check_readability(sections: dict) -> list:
    """Run readability checks. Returns list of (section, metric, value, status, threshold)."""
    results = []
    for name, text in sorted(sections.items()):
        is_main = any(name.startswith(s) for s in MAIN_SECTIONS) or name == "abstract"
        is_abstract = name == "abstract"
        fre_min = THRESHOLDS["flesch_main_min"] if is_main else THRESHOLDS["flesch_appendix_min"]
        fre_max = THRESHOLDS["flesch_main_max"] if is_main else 80

        sentences = split_sentences(text)
        if not sentences:
            continue
        word_counts = [len(s.split()) for s in sentences]
        median_wc = sorted(word_counts)[len(word_counts) // 2] if word_counts else 0
        long_40 = sum(1 for c in word_counts if c > 40)
        long_60 = sum(1 for c in word_counts if c > 60)

        fre = flesch_reading_ease(text)

        # Passive voice clusters
        passive_re = re.compile(r"\b(?:was|is|were|are|be|been|being)\s+\w+(?:ed|en)\b", re.IGNORECASE)
        consecutive_passive = 0
        max_passive_cluster = 0
        for s in sentences:
            if passive_re.search(s):
                consecutive_passive += 1
                max_passive_cluster = max(max_passive_cluster, consecutive_passive)
            else:
                consecutive_passive = 0

        # Hedging
        hedge_re = re.compile(r"\b(?:may be|might be|could be|possibly|it appears|it seems|arguably|presumably)\b", re.IGNORECASE)
        hedge_count = sum(1 for s in sentences if hedge_re.search(s))
        hedge_ratio = hedge_count / len(sentences) if sentences else 0

        # Flesch
        fre_ok = fre_min <= fre <= fre_max
        fre_status = "PASS" if fre_ok else "FAIL"
        fre_threshold = f"[{fre_min}, {fre_max}]"
        results.append((name, "flesch", f"{fre:.1f}", fre_status, fre_threshold))

        # Median sentence length
        med_ok = THRESHOLDS["median_sentence_min"] <= median_wc <= THRESHOLDS["median_sentence_max"]
        results.append((name, "median_words", str(median_wc), "PASS" if med_ok else "FAIL",
                        f"[{THRESHOLDS['median_sentence_min']}, {THRESHOLDS['median_sentence_max']}]"))

        # Long sentences
        l40_ok = long_40 <= THRESHOLDS["sentences_gt40_max"]
        results.append((name, "sentences_gt40", str(long_40), "PASS" if l40_ok else "FAIL",
                        f"<= {THRESHOLDS['sentences_gt40_max']}"))

        l60_ok = long_60 <= THRESHOLDS["sentences_gt60_max"]
        results.append((name, "sentences_gt60", str(long_60), "PASS" if l60_ok else "FAIL",
                        f"<= {THRESHOLDS['sentences_gt60_max']}"))

        # Passive clusters
        p_ok = max_passive_cluster <= THRESHOLDS["passive_cluster_max"]
        results.append((name, "passive_cluster", str(max_passive_cluster), "PASS" if p_ok else "FAIL",
                        f"<= {THRESHOLDS['passive_cluster_max']}"))

        # Hedge ratio
        h_ok = hedge_ratio <= THRESHOLDS["hedge_per_sentence_max"]
        results.append((name, "hedge_ratio", f"{hedge_ratio:.2f}", "PASS" if h_ok else "FAIL",
                        f"<= {THRESHOLDS['hedge_per_sentence_max']}"))

    return results


# ─── density metrics ──────────────────────────────────────────────────────

def count_numbers(text: str) -> int:
    """Count standalone numbers (integers, decimals, not in math)."""
    return len(re.findall(r"\b\d+\.?\d*\b", text))


def count_brackets(text: str) -> int:
    """Count parenthetical expressions, excluding citation groups."""
    brackets = re.findall(r"\([^)]+\)", text)
    # Exclude citation-like brackets: (Author~, Author~) or (Author et al.)
    return sum(1 for b in brackets if not re.search(r"~[,)]|[A-Z][a-z]+\s+(et al|&)", b))


def check_density(sections: dict) -> list:
    """Run density checks. Returns list of (section, metric, value, status, threshold)."""
    results = []
    total_numbers = 0
    total_words = 0
    total_brackets = 0

    for name, text in sorted(sections.items()):
        words = text.split()
        wc = len(words)
        if wc == 0:
            continue
        numbers = count_numbers(text)
        brackets = count_brackets(text)

        num_density = numbers / wc * 100
        brk_density = brackets / wc * 100
        words_per_num = wc / numbers if numbers > 0 else 999

        is_main = any(name.startswith(s) for s in MAIN_SECTIONS) or name == "abstract"

        # Number density
        nd_threshold = THRESHOLDS["number_density_main_max"] if is_main else THRESHOLDS["number_density_abs_max"]
        nd_ok = num_density <= nd_threshold
        results.append((name, "num_density", f"{num_density:.1f}", "PASS" if nd_ok else "FAIL",
                        f"<= {nd_threshold}"))

        # Bracket density
        bd_threshold = THRESHOLDS["bracket_density_main_max"] if is_main else 5.0
        bd_ok = brk_density <= bd_threshold
        results.append((name, "brk_density", f"{brk_density:.1f}", "PASS" if bd_ok else "FAIL",
                        f"<= {bd_threshold}"))

        # Words per number
        wpn_ok = words_per_num >= THRESHOLDS["words_per_number_min"] or not is_main
        results.append((name, "words_per_num", f"{words_per_num:.0f}", "PASS" if wpn_ok else "FAIL",
                        f">= {THRESHOLDS['words_per_number_min']}"))

        # Abstract-specific
        if name == "abstract":
            abs_ok = numbers <= THRESHOLDS["abstract_numbers_max"]
            results.append((name, "abstract_numbers", str(numbers), "PASS" if abs_ok else "FAIL",
                            f"<= {THRESHOLDS['abstract_numbers_max']}"))

        if is_main:
            total_numbers += numbers
            total_words += wc
            total_brackets += brackets

    # Aggregate
    agg_nd = total_numbers / total_words * 100 if total_words else 0
    agg_bd = total_brackets / total_words * 100 if total_words else 0
    agg_wpn = total_words / total_numbers if total_numbers else 999

    agg_nd_ok = agg_nd <= THRESHOLDS["number_density_main_max"]
    results.append(("MAIN_TOTAL", "num_density", f"{agg_nd:.1f}", "PASS" if agg_nd_ok else "FAIL",
                    f"<= {THRESHOLDS['number_density_main_max']}"))

    agg_bd_ok = agg_bd <= THRESHOLDS["bracket_density_main_max"]
    results.append(("MAIN_TOTAL", "brk_density", f"{agg_bd:.1f}", "PASS" if agg_bd_ok else "FAIL",
                    f"<= {THRESHOLDS['bracket_density_main_max']}"))

    agg_wpn_ok = agg_wpn >= THRESHOLDS["words_per_number_min"]
    results.append(("MAIN_TOTAL", "words_per_num", f"{agg_wpn:.0f}", "PASS" if agg_wpn_ok else "FAIL",
                    f">= {THRESHOLDS['words_per_number_min']}"))

    return results


# ─── reporting ────────────────────────────────────────────────────────────

def print_table(results: list, title: str):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  {'Section':<25} {'Metric':<20} {'Value':>8}  {'Status':<6} {'Threshold'}")
    print(f"  {'-' * 75}")
    fails = 0
    for section, metric, value, status, threshold in results:
        marker = "  " if status == "PASS" else ">>"
        print(f"{marker} {section:<25} {metric:<20} {value:>8}  {status:<6} {threshold}")
        if status == "FAIL":
            fails += 1
    print(f"  {'-' * 75}")
    print(f"  {'PASS' if fails == 0 else f'{fails} FAIL'}: {len(results) - fails}/{len(results)} checks")
    return fails


def main():
    args = sys.argv[1:]
    do_read = "--read" in args or "--json" in args or not args
    do_density = "--density" in args or "--json" in args or not args
    as_json = "--json" in args

    sections = get_sections()

    all_results = []
    total_fails = 0

    if do_read:
        r = check_readability(sections)
        all_results.extend(r)
        if not as_json:
            total_fails += print_table(r, "READABILITY")

    if do_density:
        d = check_density(sections)
        all_results.extend(d)
        if not as_json:
            total_fails += print_table(d, "EXPRESSION DENSITY")

    if as_json:
        output = {
            "thresholds": THRESHOLDS,
            "results": [
                {"section": s, "metric": m, "value": v, "status": st, "threshold": t}
                for s, m, v, st, t in all_results
            ],
            "total_checks": len(all_results),
            "total_fails": sum(1 for _, _, _, st, _ in all_results if st == "FAIL"),
        }
        print(json.dumps(output, indent=2))
        sys.exit(0 if output["total_fails"] == 0 else 1)

    print(f"\n{'=' * 80}")
    if total_fails == 0:
        print("  ALL CHECKS PASS")
    else:
        print(f"  {total_fails} CHECK(S) FAILED")
    print(f"{'=' * 80}")
    sys.exit(0 if total_fails == 0 else 1)


if __name__ == "__main__":
    main()
