import argparse
import os
from typing import Optional
import pandas as pd


def resolve_columns(df: pd.DataFrame):
    """
    Auto-detects and standardizes target and demographic intersection columns.
    """
    cols = df.columns.tolist()
    col_map = {c.lower(): c for c in cols}

    # 1. Resolve Target Column
    target_col = None
    for candidate in ["target", "label", "is_fake", "class"]:
        if candidate in col_map:
            target_col = col_map[candidate]
            break

    if target_col is None:
        raise KeyError(
            f"Could not identify a target column in CSV. Found columns: {cols}. "
            f"Expected one of ['Target', 'target', 'label', 'is_fake']."
        )

    # 2. Resolve Intersection / Demographic Column
    intersection_col = None
    for candidate in ["intersection", "subgroup", "demographic_group"]:
        if candidate in col_map:
            intersection_col = col_map[candidate]
            break

    # If not present as a single column, construct it from gender and skin_tone
    if intersection_col is None:
        gender_col = col_map.get("gender") or col_map.get("sex")
        skin_tone_col = col_map.get("skin_tone") or col_map.get("mst") or col_map.get("monk")

        if gender_col and skin_tone_col:
            df["Intersection"] = df[gender_col].astype(str) + "_MST_" + df[skin_tone_col].astype(str)
            intersection_col = "Intersection"
        elif skin_tone_col:
            df["Intersection"] = "MST_" + df[skin_tone_col].astype(str)
            intersection_col = "Intersection"
        elif gender_col:
            df["Intersection"] = df[gender_col].astype(str)
            intersection_col = "Intersection"
        else:
            raise KeyError(
                f"Could not find demographic columns in CSV. Found: {cols}. "
                f"Expected 'Intersection' or 'gender' and 'skin_tone'."
            )

    return target_col, intersection_col


def create_balanced_subset(
    csv_path: str,
    output_csv_path: str,
    samples_per_intersection: int = 500,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Subsets the AI-Face / Deepfake CSV ensuring balanced coverage across:
    - Target (Real vs Fake)
    - Demographic Intersectional Groups (e.g. Monk Skin Tone + Gender)
    """
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Annotation file not found at: '{csv_path}'")
        print(f"  --> Please verify the file path or place your dataset CSV in the '{os.path.dirname(csv_path) or '.'}' directory.")
        return None

    print(f"\nLoading: {csv_path}...")
    df = pd.read_csv(csv_path)
    target_col, intersection_col = resolve_columns(df)

    print(f"Sampling up to {samples_per_intersection} samples per ({target_col}, {intersection_col}) group...")

    # Robust group sampling across Target and Intersection demographic groups
    chunks = []
    for _, grp in df.groupby([target_col, intersection_col]):
        n = min(len(grp), samples_per_intersection)
        chunks.append(grp.sample(n=n, random_state=seed))

    if chunks:
        subset_df = pd.concat(chunks, ignore_index=True)
    else:
        subset_df = pd.DataFrame(columns=df.columns)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save the subset annotations
    subset_df.to_csv(output_csv_path, index=False)
    print(f"--> Created subset: {len(subset_df)} samples out of {len(df)} total.")
    print(f"--> Saved subset metadata to: {output_csv_path}")

    # Summary table
    if not subset_df.empty:
        breakdown = subset_df.groupby([intersection_col, target_col]).size().unstack(fill_value=0)
        print("\nSubset Group Breakdown:")
        print(breakdown)

    return subset_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced demographic subset from dataset annotations.")
    parser.add_argument("--input_csv", type=str, default="dataset/train.csv", help="Path to input annotations CSV")
    parser.add_argument("--output_csv", type=str, default="dataset/train_subset.csv", help="Path to output subset CSV")
    parser.add_argument("--samples_per_intersection", type=int, default=500, help="Max samples per demographic intersection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # If run with default arguments, check if train.csv or mock dataset exists
    if args.input_csv == "dataset/train.csv" and not os.path.exists("dataset/train.csv"):
        if os.path.exists("data/mock_dataset/annotations.csv"):
            print("Note: 'dataset/train.csv' not found. Testing on 'data/mock_dataset/annotations.csv'...")
            create_balanced_subset(
                "data/mock_dataset/annotations.csv",
                "data/mock_dataset/annotations_subset.csv",
                samples_per_intersection=5,
                seed=args.seed,
            )
        else:
            print(f"[Notice] 'dataset/train.csv' not found. Specify your CSV with: python data/create_subset.py --input_csv <path_to_csv>")
    else:
        create_balanced_subset(
            args.input_csv,
            args.output_csv,
            samples_per_intersection=args.samples_per_intersection,
            seed=args.seed,
        )