#!/usr/bin/env bash
# resources/gene_id_maps/build_entrez_id_map.sh
#
# Build a minimal Entrez Gene ID -> gene symbol mapping (human) from NCBI gene_info.
#
# Outputs:
#   resources/gene_id_maps/id_map.tsv.gz
#   resources/gene_id_maps/checksums.sha256
#   resources/gene_id_maps/Homo_sapiens.gene_info    (downloaded + decompressed)
#
# Usage:
#   bash resources/gene_id_maps/build_entrez_id_map.sh
#   # optional
#   bash resources/gene_id_maps/build_entrez_id_map.sh --outdir resources/gene_id_maps
#   bash resources/gene_id_maps/build_entrez_id_map.sh --url <custom_gene_info_gz_url>
#
# Notes:
# - Deterministic and offline-friendly once downloaded.
# - Keeps the header line: gene_id<TAB>gene_symbol
# - Uses cut on columns 2,3 (GeneID, Symbol). The NCBI file is human-only in this URL.
# - If you want to strictly enforce tax_id==9606, use the awk variant (see README suggestion).

set -euo pipefail

URL_DEFAULT="https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
URL="$URL_DEFAULT"
OUTDIR="resources/gene_id_maps"

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

# -------------------------
# args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)
      OUTDIR="${2:-}"; [[ -n "$OUTDIR" ]] || die "--outdir requires a value"
      shift 2
      ;;
    --url)
      URL="${2:-}"; [[ -n "$URL" ]] || die "--url requires a value"
      shift 2
      ;;
    -h|--help)
      sed -n '1,120p' "$0"
      exit 0
      ;;
    *)
      die "Unknown arg: $1"
      ;;
  esac
done

# -------------------------
# deps
# -------------------------
need_cmd gunzip || die "gunzip not found (install gzip)."
need_cmd gzip   || die "gzip not found (install gzip)."
need_cmd cut    || die "cut not found (coreutils)."
need_cmd sha256sum || die "sha256sum not found (macOS: brew install coreutils)."

# Prefer wget, fall back to curl
fetch() {
  local url="$1"
  local out="$2"

  if need_cmd wget; then
    wget -q --show-progress -O "$out" "$url"
    return 0
  fi
  if need_cmd curl; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
    return 0
  fi
  die "Neither wget nor curl is available."
}

mkdir -p "$OUTDIR"

RAW_GZ="$OUTDIR/Homo_sapiens.gene_info.gz"
RAW_TXT="$OUTDIR/Homo_sapiens.gene_info"
OUT_TSV="$OUTDIR/id_map.tsv"
OUT_GZ="$OUTDIR/id_map.tsv.gz"
CHKS="$OUTDIR/checksums.sha256"

echo "[build_entrez_id_map] Downloading: $URL" >&2
fetch "$URL" "$RAW_GZ"

echo "[build_entrez_id_map] Decompressing: $(basename "$RAW_GZ")" >&2
gunzip -f "$RAW_GZ"  # produces $RAW_TXT

[[ -s "$RAW_TXT" ]] || die "Decompressed file is empty: $RAW_TXT"

echo "[build_entrez_id_map] Writing mapping TSV..." >&2
# Keep header (optional): gene_id<TAB>gene_symbol
{
  echo -e "gene_id\tgene_symbol"
  cut -f2,3 "$RAW_TXT"
} > "$OUT_TSV"

# Basic sanity: at least some rows
n_lines="$(wc -l < "$OUT_TSV" | tr -d '[:space:]')"
[[ "${n_lines:-0}" -ge 1000 ]] || die "Mapping TSV seems too small (lines=$n_lines). Check input."

echo "[build_entrez_id_map] Compressing: $(basename "$OUT_TSV")" >&2
gzip -f "$OUT_TSV"  # produces $OUT_GZ

echo "[build_entrez_id_map] Writing checksums..." >&2
# Hash only deliverables (avoid hashing the big raw file by default)
(
  cd "$OUTDIR"
  sha256sum "$(basename "$OUT_GZ")" > "$(basename "$CHKS")"
)

echo "OK"
echo "  - $OUT_GZ"
echo "  - $CHKS"
echo ""
echo "SampleCard config example:"
echo "  extra:"
echo "    gene_id_map_tsv: \"$OUT_GZ\""
