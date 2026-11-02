#!/usr/bin/env bash
# Extract matching AlphaFold v6 .pdb.gz entries from a local AFDB bulk tarball, gunzip them,
# and join them onto an id/seq/ec_number CSV to produce build_structures.py's --input-csv.
#
# Requires a local AFDB v6 bulk tarball (e.g. "swissprot_pdb_v6.tar" from the UniProt/Swiss-Prot
# bulk download at https://alphafold.ebi.ac.uk/download). This script does not fetch it.
#
# Usage:
#   bash preprocessing/swissprot_ec/download_alphafold_pdbs.sh \
#       <input_csv> <tarball_path> <out_dir> [python_bin]
#
# <input_csv> columns: id, seq, ec_number (see preprocessing/swissprot_ec/README.md).
#
# Outputs (under <out_dir>):
#   alphafold_pdbs/              gunzipped PDB files for matched ids
#   train_100_with_pdb.csv       input CSV + a pdb_path column, relative to <out_dir>

set -euo pipefail

INPUT_CSV="${1:?usage: download_alphafold_pdbs.sh <input_csv> <tarball_path> <out_dir> [python_bin]}"
TARBALL_PATH="${2:?missing <tarball_path>}"
OUT_DIR="${3:?missing <out_dir>}"
PYTHON="${4:-python3}"

PDB_DIR="${OUT_DIR}/alphafold_pdbs"
OUTPUT_CSV="${OUT_DIR}/train_100_with_pdb.csv"

mkdir -p "$PDB_DIR"

echo "[1/3] Building file list from ${INPUT_CSV}..."
FILE_LIST=$(mktemp)
tail -n +2 "$INPUT_CSV" | cut -d',' -f1 | sort -u | while read -r uid; do
    echo "AF-${uid}-F1-model_v6.pdb.gz"
done > "$FILE_LIST"
echo "       $(wc -l < "$FILE_LIST") files to extract."

echo "[2/3] Extracting matched .pdb.gz from tarball and gunzipping..."
tar -xf "$TARBALL_PATH" -C "$PDB_DIR" --files-from="$FILE_LIST" 2>&1 | tail -5 || true
find "$PDB_DIR" -name "*.pdb.gz" -exec gunzip -f {} +
rm -f "$FILE_LIST"

EXTRACTED=$(find "$PDB_DIR" -name "*.pdb" | wc -l)
echo "       ${EXTRACTED} PDB files ready."

echo "[3/3] Building output CSV: ${OUTPUT_CSV}"
"$PYTHON" - "$INPUT_CSV" "alphafold_pdbs" "$PDB_DIR" "$OUTPUT_CSV" <<'PYEOF'
import csv
import os
import sys

input_csv, pdb_rel_dir, pdb_dir, output_csv = sys.argv[1:5]

# Index available PDB files: AF-P00958-F1-model_v6.pdb -> P00958. pdb_path is written
# relative to --out-data-dir / --pdb-root, matching build_structures.py's convention.
available = {}
for fname in os.listdir(pdb_dir):
    if fname.startswith("AF-") and fname.endswith(".pdb"):
        uid = fname.split("-")[1]
        available[uid] = f"{pdb_rel_dir}/{fname}"

matched = missed = 0
with open(input_csv) as fin, open(output_csv, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["id", "seq", "ec_number", "pdb_path"])
    writer.writeheader()
    for row in reader:
        uid = row["id"]
        if uid in available:
            writer.writerow({"id": uid, "seq": row["seq"], "ec_number": row["ec_number"],
                              "pdb_path": available[uid]})
            matched += 1
        else:
            missed += 1

print(f"  Matched: {matched}")
print(f"  Missed:  {missed}")
if matched + missed:
    print(f"  Coverage: {matched / (matched + missed) * 100:.2f}%")
print(f"  Output:  {output_csv}")
PYEOF

echo ""
echo "Done."
echo "  PDB files:  ${PDB_DIR}/"
echo "  Output CSV: ${OUTPUT_CSV}"
echo "  (pass --pdb-root <out_dir> to build_structures.py so pdb_path resolves)"
