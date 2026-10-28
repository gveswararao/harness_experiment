"""
Combines results.csv files from separate model runs into a single file for
cross-model analysis (analyze_results.py / compute_icc.py both expect one
combined results.csv with a 'model' column distinguishing rows).

Usage:
    python merge_results.py results.csv results_phi4mini.csv --output results_combined.csv
"""
import argparse
import csv


def main(inputs, output_path):
    all_rows = []
    fieldnames = None
    for path in inputs:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)
        print(f"Read {path}")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    models = sorted(set(r["model"] for r in all_rows))
    print(f"\nWrote {len(all_rows)} total rows to {output_path}")
    print(f"Models included: {models}")
    for m in models:
        n = sum(1 for r in all_rows if r["model"] == m)
        print(f"  {m}: {n} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Two or more results CSV files to combine")
    parser.add_argument("--output", type=str, default="results_combined.csv")
    args = parser.parse_args()
    main(args.inputs, args.output)
