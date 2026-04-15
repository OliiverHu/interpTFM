import os, glob
import pandas as pd

RUNS_ROOT = "runs/full_scgpt_cosmx"
HELDOUT_ROOT = os.path.join(RUNS_ROOT, "heldout_report")

PATTERNS = [
    "**/*.csv",
    "**/*.tsv",
    "**/*.parquet",
]

def try_read(fp: str):
    if fp.endswith(".parquet"):
        return pd.read_parquet(fp)
    if fp.endswith(".tsv"):
        return pd.read_csv(fp, sep="\t")
    return pd.read_csv(fp)

def main():
    files = []
    for pat in PATTERNS:
        files.extend(glob.glob(os.path.join(HELDOUT_ROOT, pat), recursive=True))
    files = sorted(set(files))

    print("HELDOUT_ROOT:", HELDOUT_ROOT)
    print("n_files:", len(files))
    for i, fp in enumerate(files[:50]):
        print(f"[{i:02d}] {fp}")

    if not files:
        print("No files found. That means heldout_report_for_layer didn't write anything here.")
        return

    # Peek at first ~10 readable tabular files
    shown = 0
    for fp in files:
        if not (fp.endswith(".csv") or fp.endswith(".tsv") or fp.endswith(".parquet")):
            continue
        try:
            df = try_read(fp)
        except Exception as e:
            print("\n---")
            print("FAILED READ:", fp)
            print("ERROR:", e)
            continue

        print("\n---")
        print("FILE:", fp)
        print("SHAPE:", df.shape)
        print("COLUMNS:", list(df.columns))
        print(df.head(3).to_string(index=False))

        shown += 1
        if shown >= 10:
            break

if __name__ == "__main__":
    main()