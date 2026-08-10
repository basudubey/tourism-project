"""
Data Preparation
-----------------
Loads the dataset from the repository data folder, cleans it (dedupes
inconsistent category labels, engineers an AgeGroup feature, validates
CustomerID uniqueness and null counts), removes unnecessary columns, and
splits it into train/test sets. The splits are saved locally as CSV files
so the workflow can pass them to the next job as an artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

TARGET = "ProdTaken"

# Columns that carry no predictive signal (pure identifiers)
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]

AGE_BINS = [17, 25, 35, 45, 55, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56+"]


def normalize_occupation(val: str) -> str:
    """Collapse spacing/casing variants (e.g. 'Freelancer', 'free lance',
    'Free Lancer') down to a single canonical label."""
    key = str(val).strip().lower().replace(" ", "")
    mapping = {
        "freelancer": "Free Lancer",
        "freelance": "Free Lancer",
        "salaried": "Salaried",
        "smallbusiness": "Small Business",
        "largebusiness": "Large Business",
    }
    return mapping.get(key, str(val).strip())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # --- CustomerID must be a unique key before we drop it ---
    n_dupes = df["CustomerID"].duplicated().sum()
    if n_dupes:
        raise ValueError(f"CustomerID is not unique: {n_dupes} duplicate id(s) found.")
    print(f"CustomerID uniqueness check passed ({df['CustomerID'].nunique()} unique ids).")

    # --- Null check (report only; this dataset has none, but flag if that changes) ---
    null_counts = df.isnull().sum()
    if null_counts.sum():
        print("Null values found per column:")
        print(null_counts[null_counts > 0])
    else:
        print("No null values found.")

    # --- Drop exact duplicate rows, if any ---
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate row(s).")

    # --- Fix inconsistent category labels found during EDA ---
    # Gender: 'Fe Male' -> 'Female'
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

    # Occupation: normalize 'Free Lancer' / 'Freelancer' / 'free lance' variants
    df["Occupation"] = df["Occupation"].apply(normalize_occupation)

    # MaritalStatus: 'Single' and 'Unmarried' mean the same thing -> merge into 'Single'
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # --- Feature engineering: bucket Age into groups ---
    df["AgeGroup"] = pd.cut(df["Age"], bins=AGE_BINS, labels=AGE_LABELS)

    # --- Drop identifier columns now that validation is done ---
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = clean(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("\nData preparation complete.")
    print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
    print(f"ytrain: {ytrain.shape}, ytest: {ytest.shape}")
    print(f"\nOccupation values after cleaning: {sorted(X['Occupation'].unique())}")
    print(f"MaritalStatus values after cleaning: {sorted(X['MaritalStatus'].unique())}")
    print(f"AgeGroup distribution:\n{X['AgeGroup'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
