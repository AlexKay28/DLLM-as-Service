"""
Download the four LOGHUB / USENIX CFDR corpora used by the FlowFill paper.

Corpora:
  - HDFS        (LOGHUB / Zenodo)        ~178 MiB  11,175,629 lines (uncompressed)
  - BGL         (LOGHUB / Zenodo)         ~71 MiB   4,747,963 lines
  - Thunderbird (LOGHUB / Zenodo)         ~29 GiB  211,212,192 lines
  - Spirit      (USENIX CFDR HPC4)       ~864 MiB ~272,000,000 lines

Usage:
    python download.py                       # download all four to datasets/raw/
    python download.py --only hdfs bgl       # download a subset
    python download.py --raw-dir /data/logs  # custom destination
    python download.py --verify              # only verify existing archives' sizes
    python download.py --samples             # also fetch the 2k LOGHUB samples for smoke tests

Design:
  - Pure stdlib (urllib + hashlib) so it runs anywhere; no pip install.
  - Resumable: appends to a `.part` file and resumes via HTTP Range on retry.
  - Verifies the downloaded size against the server's Content-Length (the
    authoritative source) rather than hardcoded values.
  - Never overwrites a complete archive; skip if the server-reported size
    is already on disk.

Note: Zenodo and the USENIX CDN both support HTTP Range requests, so resumption
is safe. Thunderbird and Spirit are very large (29 GiB / 864 MiB); use --only
to fetch a subset if bandwidth is limited.
"""

import argparse
import shutil
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "datasets" / "raw"

# Approximate compressed sizes for *human-friendly* progress reporting only.
# These are NOT used for verification — we verify against the server-reported
# Content-Length at download time, which is authoritative. Sizes sourced from
# the Zenodo record 8196385 file listing and the USENIX HPC4 listing.
APPROX_SIZES = {
    "hdfs":        186 * (1 << 20),   # HDFS_v1.zip        ~178 MiB (compressed)
    "bgl":          71 * (1 << 20),   # BGL.zip             ~71 MiB
    "thunderbird": 29 * (1 << 30),   # Thunderbird.tar.gz ~29 GiB
    "spirit":      864 * (1 << 20),   # spirit2.gz         ~864 MiB
}

ARCHIVE_NAMES = {
    "hdfs":        "HDFS_v1.zip",
    "bgl":         "BGL.zip",
    "thunderbird": "Thunderbird.tar.gz",
    "spirit":      "Spirit.gz",
}

URLS = {
    "hdfs":        "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1",
    "bgl":         "https://zenodo.org/records/8196385/files/BGL.zip?download=1",
    "thunderbird": "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1",
    "spirit":      "http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz",
}

# 2k-line samples from the logpai/loghub GitHub repo (raw master).
SAMPLE_URLS = {
    "hdfs":        "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log",
    "bgl":         "https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log",
    "thunderbird": "https://raw.githubusercontent.com/logpai/loghub/master/Thunderbird/Thunderbird_2k.log",
}
SAMPLE_NAMES = {
    "hdfs":        "HDFS_2k.log",
    "bgl":         "BGL_2k.log",
    "thunderbird": "Thunderbird_2k.log",
}

CHUNK = 1 << 16  # 64 KiB
TIMEOUT = 60      # seconds per connect/read


@dataclass
class DownloadTask:
    name: str
    url: str
    dest: Path
    approx_size: int | None  # only for progress display; not for verification


def _fmt_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{n} B"
        f /= 1024.0
    return f"{f:.1f} TiB"


def _remote_size(url: str) -> int | None:
    """Get Content-Length via a HEAD request; return None if not reported."""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "DLLM-as-Service/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except (urllib.error.URLError, ValueError):
        return None


def _content_length(resp) -> int | None:
    cl = resp.headers.get("Content-Length")
    if not cl:
        return None
    try:
        return int(cl)
    except ValueError:
        return None


def _download_resumable(task: DownloadTask) -> tuple[bool, str]:
    """Download `task` to `task.dest`, resuming from any existing `.part` file.

    Verification is against the server's Content-Length (authoritative).
    Returns (ok, message).
    """
    part_path = task.dest.with_suffix(task.dest.suffix + ".part")
    have = part_path.stat().st_size if part_path.exists() else 0

    headers = {"User-Agent": "DLLM-as-Service/1.0"}
    if have:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(task.url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            # If server ignored Range, restart from scratch.
            if have and resp.status == 200:
                have = 0
                mode = "wb"
            elif have and resp.status == 206:
                mode = "ab"
            else:
                have = 0
                mode = "wb"

            # For 206 responses Content-Length is the remaining bytes; for 200
            # it's the full size. Use the approximation only if absent.
            remaining = _content_length(resp)
            if resp.status == 206 and remaining is not None and have:
                total = have + remaining
            else:
                total = remaining or task.approx_size

            with open(part_path, mode) as f:
                downloaded = have
                last_report = 0
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded - last_report >= (1 << 20):
                        pct = 100.0 * downloaded / total
                        print(f"    {task.name:11s} {pct:5.1f}%  "
                              f"{_fmt_bytes(downloaded)}/{_fmt_bytes(total)}",
                              end="\r", flush=True)
                        last_report = downloaded
        print()
    except urllib.error.URLError as e:
        return False, f"network error: {e}"

    # Validate size against the server-reported total if we have it.
    got = part_path.stat().st_size
    if total and got != total:
        return False, (f"size mismatch: got {got}, expected {total} "
                       "(kept .part for resume on retry)")
    # Rename to final destination.
    shutil.move(str(part_path), str(task.dest))
    return True, f"done ({_fmt_bytes(got)})"


def _verify_existing(task: DownloadTask, server_size: int | None
                     ) -> tuple[bool, str]:
    if not task.dest.exists():
        return False, "missing"
    size = task.dest.stat().st_size
    if server_size and size != server_size:
        return False, f"size mismatch: {size} != {server_size} (server)"
    if server_size:
        return True, f"ok ({_fmt_bytes(size)})"
    return True, f"present ({_fmt_bytes(size)}, server size unknown)"


def _extract_archive(task: DownloadTask, raw_dir: Path) -> str:
    """Extract `task.dest` into a sibling directory named after the corpus.

    Returns the path to the extracted directory.
    """
    out_dir = raw_dir / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if task.dest.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(task.dest) as zf:
            zf.extractall(out_dir)
    elif task.dest.suffix == ".gz":
        # .gz that is a single-file compressed log (spirit2.gz -> Spirit)
        target = out_dir / "Spirit.log"
        import gzip
        with gzip.open(task.dest, "rb") as gz, open(target, "wb") as out:
            shutil.copyfileobj(gz, out)
    elif task.dest.suffix == ".tgz" or task.dest.name.endswith(".tar.gz"):
        import tarfile
        with tarfile.open(task.dest) as tf:
            tf.extractall(out_dir)
    return str(out_dir)


def _fetch_sample(name: str, raw_dir: Path) -> tuple[bool, str]:
    """Fetch a 2k LOGHUB sample for smoke testing."""
    url = SAMPLE_URLS[name]
    dest = raw_dir / name / SAMPLE_NAMES[name]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return True, f"already present ({dest.stat().st_size} B)"
    req = urllib.request.Request(url, headers={"User-Agent": "DLLM-as-Service/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, \
                open(dest, "wb") as f:
            shutil.copyfileobj(resp, f)
    except urllib.error.URLError as e:
        return False, f"network error: {e}"
    return True, f"ok ({dest.stat().st_size} B)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR),
                        help=f"Destination dir (default: {DEFAULT_RAW_DIR})")
    parser.add_argument("--only", nargs="+", choices=list(URLS.keys()),
                        help="Download only these corpora")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify already-downloaded archives; skip fetching")
    parser.add_argument("--extract", action="store_true",
                        help="Extract archives after a successful download")
    parser.add_argument("--samples", action="store_true",
                        help="Also fetch the 2k LOGHUB samples for smoke tests")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if a complete archive is present")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    targets = args.only or list(URLS.keys())

    print(f"Raw dir: {raw_dir}")
    print(f"Targets: {', '.join(targets)}")
    print()

    # Verify-only mode.
    if args.verify:
        print("== Verify mode ==")
        all_ok = True
        for name in targets:
            task = DownloadTask(name, URLS[name], raw_dir / ARCHIVE_NAMES[name],
                                APPROX_SIZES.get(name))
            server_size = _remote_size(task.url)
            ok, msg = _verify_existing(task, server_size)
            extra = f" (server: {_fmt_bytes(server_size)})" if server_size else ""
            print(f"  {name:11s} {msg}{extra}")
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    # Fetch samples if requested.
    if args.samples:
        print("== 2k samples ==")
        for name in SAMPLE_URLS:
            ok, msg = _fetch_sample(name, raw_dir)
            print(f"  {name:11s} {msg}")
        print()

    # Download / verify archives.
    print("== Corpora ==")
    failures = []
    for name in targets:
        dest = raw_dir / ARCHIVE_NAMES[name]
        task = DownloadTask(name, URLS[name], dest, APPROX_SIZES.get(name))
        print(f"\n[{name}] -> {dest.name}")
        print(f"  url: {task.url}")

        # Probe server size once for skip/verify decisions.
        server_size = _remote_size(task.url)

        if dest.exists() and not args.force:
            ok, msg = _verify_existing(task, server_size)
            print(f"  already present: {msg}")
            if ok and args.extract:
                path = _extract_archive(task, raw_dir)
                print(f"  extracted to {path}")
            if not ok:
                failures.append(name)
            continue

        ok, msg = _download_resumable(task)
        print(f"  {msg}")
        if ok and args.extract:
            path = _extract_archive(task, raw_dir)
            print(f"  extracted to {path}")
        if not ok:
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("All requested corpora ready.")


if __name__ == "__main__":
    main()
