import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def collapse_rare_categories(series: pd.Series, min_count: int = 20) -> pd.Series:
    """
    Replace low-frequency categories with 'Other' to reduce sparsity after one-hot encoding.
    """
    counts = series.value_counts(dropna=False)
    rare = counts[counts < min_count].index
    return series.where(~series.isin(rare), other="Other")


def split_train_tune_test(X, y, test_size=0.20, tune_size=0.20, random_state=42, stratify=None):
    """
    Returns: X_train, X_tune, X_test, y_train, y_tune, y_test
    Example: test_size=0.20 and tune_size=0.20 yields 60/20/20.
    """
    # First split off TEST
    X_train_tune, X_test, y_train_tune, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Now split train_tune into TRAIN and TUNE
    # tune_size is proportion of original dataset; convert to proportion of the remaining chunk:
    tune_prop_of_remaining = tune_size / (1 - test_size)

    stratify_2 = y_train_tune if stratify is not None else None
    X_train, X_tune, y_train, y_tune = train_test_split(
        X_train_tune,
        y_train_tune,
        test_size=tune_prop_of_remaining,
        random_state=random_state,
        stratify=stratify_2
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test


def build_preprocessor(X: pd.DataFrame):
    """
    Detect categorical vs numeric columns and build a ColumnTransformer:
    - one-hot encode categoricals
    - scale numerics
    """
    cat_cols = X.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    return preprocessor, cat_cols, num_cols


# -----------------------------
# Pipeline 1: College dataset
# target = student_count (regression)
# -----------------------------

def college_pipeline(path="cc_institution_details.csv", target_col="student_count", random_state=42):
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.drop_duplicates()

    # Drop rows missing target
    df = df.dropna(subset=[target_col])

    # Separate X/y
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # OPTIONAL: collapse rare categories (helps if you have columns with tons of categories)
    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[col] = collapse_rare_categories(X[col].astype(str), min_count=20)

    # Split (no stratify for regression)
    X_train, X_tune, X_test, y_train, y_tune, y_test = split_train_tune_test(
        X, y, test_size=0.20, tune_size=0.20, random_state=random_state, stratify=None
    )

    # Preprocess: fit on train, transform tune/test
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_tune_p  = preprocessor.transform(X_tune)
    X_test_p  = preprocessor.transform(X_test)

    # Sanity checks
    print("\n[College Pipeline]")
    print("Categorical cols:", cat_cols)
    print("Numeric cols:", num_cols)
    print("Shapes (train/tune/test):", X_train_p.shape, X_tune_p.shape, X_test_p.shape)
    print("Target summary (student_count):")
    print(y_train.describe())

    return X_train_p, X_tune_p, X_test_p, y_train, y_tune, y_test


# -----------------------------
# Pipeline 2: Job placement dataset
# target = status (classification)
# -----------------------------

def job_pipeline(path="job_placement.csv", target_col="status", random_state=42):
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.drop_duplicates()

    # Drop rows missing target
    df = df.dropna(subset=[target_col])

    # If there is a serial number column, drop it if present
    for maybe_id in ["sl_no", "id", "ID"]:
        if maybe_id in df.columns:
            df = df.drop(columns=[maybe_id])

    # Separate X/y
    y_raw = df[target_col].astype(str).str.strip()
    # Map to 1/0 (Placed -> 1, Not Placed -> 0)
    y = y_raw.map(lambda v: 1 if v.lower() == "placed" else 0)

    X = df.drop(columns=[target_col])
    
    # Drop salary if present (data leakage)
    if "salary" in X.columns:
        X = X.drop(columns=["salary"])

    # Collapse rare categories in categoricals
    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[col] = collapse_rare_categories(X[col].astype(str), min_count=20)

    # Stratify for classification
    X_train, X_tune, X_test, y_train, y_tune, y_test = split_train_tune_test(
        X, y, test_size=0.20, tune_size=0.20, random_state=random_state, stratify=y
    )

    # Preprocess: fit on train, transform tune/test
    preprocessor, cat_cols, num_cols = build_preprocessor(X_train)

    X_train_p = preprocessor.fit_transform(X_train)
    X_tune_p  = preprocessor.transform(X_tune)
    X_test_p  = preprocessor.transform(X_test)

    # Prevalence (class balance)
    prevalence = y_train.mean()  # since placed=1, this is % placed in training
    print("\n[Job Pipeline]")
    print("Categorical cols:", cat_cols)
    print("Numeric cols:", num_cols)
    print("Shapes (train/tune/test):", X_train_p.shape, X_tune_p.shape, X_test_p.shape)
    print(f"Training prevalence (Placed=1): {prevalence:.3f}")

    return X_train_p, X_tune_p, X_test_p, y_train, y_tune, y_test


# -----------------------------
# Run both pipelines (so grader can see it works)
# -----------------------------
if __name__ == "__main__":
    # Make sure these filenames match what you actually saved in your repo
    college_pipeline(path="cc_institution_details.csv")
    job_pipeline(path="job_placement.csv")