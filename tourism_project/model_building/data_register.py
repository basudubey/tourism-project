"""
Data Registration
------------------
Reads the tourism dataset from the repository's data folder, validates that
all expected columns are present, and prints a short summary. This acts as
a lightweight "registration" step confirming the dataset is available and
well-formed before it moves into the pipeline.
"""

import os
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def main():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    # Drop the stray index column that pandas sometimes writes/reads back
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: missing expected columns: {missing}")
        sys.exit(1)

    print("Dataset Registration Summary")
    print("=============================")
    print(f"Path              : {DATA_PATH}")
    print(f"Rows, Columns     : {df.shape}")
    print(f"Columns           : {list(df.columns)}")
    print(f"Target (ProdTaken) value counts:\n{df['ProdTaken'].value_counts()}")
    print(f"Missing values per column:\n{df.isnull().sum()}")
    print("\nAll expected columns are present. Dataset registered successfully.")


if __name__ == "__main__":
    main()
