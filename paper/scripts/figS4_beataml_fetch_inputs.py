#!/usr/bin/env python3
# paper/scripts/figS4_beataml_fetch_inputs.py
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

# =========================
# Fixed paths (v1)
# =========================
ROOT = Path(__file__).resolve().parents[1]  # paper/
SD = ROOT / "source_data" / "BEATAML_TP53_v1"
RAW = SD / "raw"


# =========================
# Public URLs (GitHub raw)
# =========================
DEFAULT_COUNTS_URL = (
    "https://github.com/biodev/beataml2.0_data/raw/main/beataml_waves1to4_counts_dbgap.txt"
)
DEFAULT_MUT_URL = (
    "https://github.com/biodev/beataml2.0_data/raw/main/beataml_wes_wv1to4_mutations_dbgap.txt"
)
DEFAULT_CLIN_URL = "https://github.com/biodev/beataml2.0_data/raw/main/beataml_wv1to4_clinical.xlsx"


def _die(msg: str) -> None:
    """
    Exit the script with an error message.

    Parameters
    ----------
    msg : str
        Error message displayed to the user.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        Always raised with the given message.
    """
    raise SystemExit(msg)


def _ensure_dir(p: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    p : pathlib.Path
        Directory path to create.

    Returns
    -------
    None
    """
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 checksum of a file.

    Parameters
    ----------
    path : pathlib.Path
        Path to an existing file.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.

    Notes
    -----
    The file is streamed in chunks to avoid loading it into memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, out_path: Path, *, timeout_sec: int = 120) -> None:
    """
    Download a URL to a local path using a temporary file and atomic replace.

    Parameters
    ----------
    url : str
        Source URL.
    out_path : pathlib.Path
        Destination file path.
    timeout_sec : int, optional
        Timeout for ``urllib.request.urlopen`` in seconds.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the download fails for any reason.

    Notes
    -----
    The download is streamed to ``<out_path>.<suffix>.tmp`` and then replaced
    atomically to avoid partial outputs on interruption.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "llm-pathway-curator/beataml-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r, tmp.open("wb") as w:
            shutil.copyfileobj(r, w)
        tmp.replace(out_path)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        _die(f"[beataml_fetch] download failed: {url}\n{e}")


def _ensure_nonempty(path: Path, *, min_bytes: int) -> None:
    """
    Validate that a file exists, is a regular file, and meets a size threshold.

    Parameters
    ----------
    path : pathlib.Path
        File path to validate.
    min_bytes : int
        Minimum allowed file size in bytes.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If the file is missing, not a file, or smaller than ``min_bytes``.
    """
    if not path.exists():
        _die(f"[beataml_fetch] missing file: {path}")
    if not path.is_file():
        _die(f"[beataml_fetch] not a file: {path}")
    n = int(path.stat().st_size)
    if n < int(min_bytes):
        _die(f"[beataml_fetch] file too small (<{min_bytes} bytes): {path} (size={n})")


def main() -> None:
    """
    CLI entry point to fetch BeatAML v2.0 raw inputs for the paper pipeline.

    Downloads fixed public inputs into:
    ``paper/source_data/BEATAML_TP53_v1/raw/``

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        If downloads fail or fetched files do not pass size sanity checks.

    Notes
    -----
    After fetching, SHA-256 checksums are printed to support provenance
    pinning and reproducible figure generation.
    """
    ap = argparse.ArgumentParser(
        description="Fetch BeatAML v2.0 raw inputs into source_data/BEATAML_TP53_v1/raw/"
    )
    ap.add_argument("--counts-url", default=DEFAULT_COUNTS_URL)
    ap.add_argument("--mutations-url", default=DEFAULT_MUT_URL)
    ap.add_argument("--clinical-url", default=DEFAULT_CLIN_URL)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    ap.add_argument(
        "--timeout-sec", type=int, default=180, help="urlopen timeout seconds (default: 180)"
    )
    # size sanity: keep conservative (these are big)
    ap.add_argument(
        "--min-bytes-counts", type=int, default=5_000_000, help="min bytes for counts file"
    )
    ap.add_argument(
        "--min-bytes-mutations", type=int, default=500_000, help="min bytes for mutations file"
    )
    ap.add_argument(
        "--min-bytes-clinical", type=int, default=50_000, help="min bytes for clinical xlsx file"
    )
    args = ap.parse_args()

    _ensure_dir(RAW)

    counts_out = RAW / "beataml_waves1to4_counts_dbgap.txt"
    muts_out = RAW / "beataml_wes_wv1to4_mutations_dbgap.txt"
    clin_out = RAW / "beataml_wv1to4_clinical.xlsx"

    plan = [
        ("counts", str(args.counts_url), counts_out, int(args.min_bytes_counts)),
        ("mutations", str(args.mutations_url), muts_out, int(args.min_bytes_mutations)),
        ("clinical", str(args.clinical_url), clin_out, int(args.min_bytes_clinical)),
    ]

    for label, url, outp, min_bytes in plan:
        if outp.exists() and (not args.force):
            print(f"[beataml_fetch] SKIP exists: {outp}", file=sys.stderr)
        else:
            print(f"[beataml_fetch] downloading {label}: {url}", file=sys.stderr)
            _download_url(url, outp, timeout_sec=int(args.timeout_sec))

        _ensure_nonempty(outp, min_bytes=min_bytes)

    print("[beataml_fetch] OK")
    for label, _, outp, _ in plan:
        print(f"  {label}: {outp}  (sha256={_sha256_file(outp)})")


if __name__ == "__main__":
    main()
