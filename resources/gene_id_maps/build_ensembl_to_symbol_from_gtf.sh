#!/usr/bin/env bash
# resources/gene_id_maps/build_ensembl_to_symbol_from_gtf.sh
#
# Build an Ensembl gene_id (ENSG...) -> gene_symbol mapping from a GTF.
#
# Default mode: OFFLINE (use a local GTF path).
# Optional mode: DOWNLOAD a GTF from a URL (Ensembl FTP), then build.
#
# Outputs:
#   resources/gene_id_maps/ensembl_id_map.tsv.gz
#   resources/gene_id_maps/checksums.sha256
#
# Usage (offline):
#   bash resources/gene_id_maps/build_ensembl_to_symbol_from_gtf.sh path/to/Homo_sapiens.GRCh38.109.gtf.gz
#   bash resources/gene_id_maps/build_ensembl_to_symbol_from_gtf.sh path/to/Homo_sapiens.GRCh38.109.gtf
#
# Usage (download):
#   bash resources/gene_id_maps/build_ensembl_to_symbol_from_gtf.sh --download
#   # or specify a custom URL:
#   bash resources/gene_id_maps/build_ensembl_to_symbol_from_gtf.sh --url <gtf_gz_url> --download
#
# Notes:
# - Restricts to feature == "gene"
# - Extracts gene_id and gene_name (falls back to gene_symbol or gene if present)
# - Drops Ensembl version suffix: ENSG... .<version>
# - Deterministic: stable sort, keep first mapping per gene_id
# - If your GTF lacks gene_name, output gene_symbol=NA for those genes.

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1; }

OUTDIR="resources/gene_id_maps"
mkdir -p "$OUTDIR"

OUT_TSV="$OUTDIR/ensembl_id_map.tsv"
OUT_GZ="$OUTDIR/ensembl_id_map.tsv.gz"
CHKS="$OUTDIR/checksums.sha256"

# Ensembl default (adjust as you like)
DEFAULT_URL="https://ftp.ensembl.org/pub/release-109/gtf/homo_sapiens/Homo_sapiens.GRCh38.109.gtf.gz"
URL="$DEFAULT_URL"
DO_DOWNLOAD=0
GTF=""

usage() {
  cat <<EOF
Usage:
  Offline:
    $0 path/to/Homo_sapiens.GRCh38.109.gtf.gz
    $0 path/to/Homo_sapiens.GRCh38.109.gtf

  Download:
    $0 --download
    $0 --download --url <gtf_gz_url>

Options:
  --download          Download GTF (gz) from --url (default: Ensembl release-109 human GTF)
  --url <URL>         Override download URL
  --outdir <DIR>      Output directory (default: $OUTDIR)
  -h, --help          Show this help
EOF
}

# -------------------------
# args
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)
      DO_DOWNLOAD=1
      shift
      ;;
    --url)
      URL="${2:-}"; [[ -n "$URL" ]] || die "--url requires a value"
      shift 2
      ;;
    --outdir)
      OUTDIR="${2:-}"; [[ -n "$OUTDIR" ]] || die "--outdir requires a value"
      shift 2
      OUT_TSV="$OUTDIR/ensembl_id_map.tsv"
      OUT_GZ="$OUTDIR/ensembl_id_map.tsv.gz"
      CHKS="$OUTDIR/checksums.sha256"
      mkdir -p "$OUTDIR"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      # positional GTF path
      if [[ -z "$GTF" ]]; then
        GTF="$1"
        shift
      else
        die "Unexpected extra arg: $1"
      fi
      ;;
  esac
done

# -------------------------
# deps
# -------------------------
need_cmd awk || die "awk not found."
need_cmd sort || die "sort not found."
need_cmd gzip || die "gzip not found."
need_cmd sha256sum || die "sha256sum not found (macOS: brew install coreutils)."

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
  die "Neither wget nor curl is available for --download."
}

if [[ "$DO_DOWNLOAD" -eq 1 ]]; then
  # download into OUTDIR
  local_gz="$OUTDIR/$(basename "$URL")"
  echo "[build_ensembl_id_map] Downloading: $URL" >&2
  fetch "$URL" "$local_gz"
  GTF="$local_gz"
fi

[[ -n "$GTF" ]] || { usage; die "Provide a GTF path or use --download."; }
[[ -f "$GTF" ]] || die "GTF not found: $GTF"

# Pick a reader
READ_CMD=(cat)
case "$GTF" in
  *.gz) need_cmd zcat || die "zcat not found (install gzip)."; READ_CMD=(zcat) ;;
esac

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

# Build map (gene_id -> gene_name / gene_symbol)
#
# Priority for symbol field:
#   gene_name > gene_symbol > gene > NA
#
# Extract:
#   gene_id "ENSG...(.version optional)"
#   gene_name "TP53" (if present)
#   gene_symbol "TP53" (rare in GTF but handle)
#   gene "TP53" (some annotations use gene instead of gene_name)
"${READ_CMD[@]}" "$GTF" \
  | awk -F'\t' 'BEGIN{OFS="\t"} $0 !~ /^#/ && $3=="gene" {print $9}' \
  | awk '
      BEGIN{OFS="\t"}
      function pick_symbol(a,b,c,  s){
        s=a; if (s=="") s=b; if (s=="") s=c; if (s=="") s="NA";
        return s;
      }
      {
        gid=""; gname=""; gsym=""; gplain="";
        if (match($0, /gene_id "[^"]+"/)) {
          gid=substr($0, RSTART+9, RLENGTH-10);
          sub(/\.[0-9]+$/, "", gid);   # drop .version
        }
        if (match($0, /gene_name "[^"]+"/)) {
          gname=substr($0, RSTART+11, RLENGTH-12);
        }
        if (match($0, /gene_symbol "[^"]+"/)) {
          gsym=substr($0, RSTART+13, RLENGTH-14);
        }
        if (match($0, /gene "[^"]+"/)) {
          gplain=substr($0, RSTART+6, RLENGTH-7);
        }
        if (gid != "") {
          print gid, pick_symbol(gname, gsym, gplain);
        }
      }
    ' \
  | LC_ALL=C sort -t $'\t' -k1,1 -k2,2 \
  | awk -F'\t' 'BEGIN{OFS="\t"} !seen[$1]++ {print $1,$2}' \
  > "$tmp"

# Write header + content
{
  echo -e "gene_id\tgene_symbol"
  cat "$tmp"
} > "$OUT_TSV"

# Basic sanity
n_lines="$(wc -l < "$OUT_TSV" | tr -d '[:space:]')"
[[ "${n_lines:-0}" -ge 1000 ]] || die "Mapping TSV seems too small (lines=$n_lines). Check input GTF."

gzip -f "$OUT_TSV"  # produces $OUT_GZ

# Hash deliverables (all *.tsv.gz in OUTDIR)
(
  cd "$OUTDIR"
  sha256sum ./*.tsv.gz > "$(basename "$CHKS")"
)

echo "OK"
echo "  - $OUT_GZ"
echo "  - $CHKS"
echo ""
echo "SampleCard config example:"
echo "  extra:"
echo "    gene_id_map_tsv: \"$OUT_GZ\""
