import numpy as np
import pandas as pd
from config import DATA_PATH, SEED


# ---- demographic subgroup definitions -----------------------
def age_group(a):
    if a < 21:  return "<21"
    if a <= 30: return "21-30"
    return "31-40"


def bmi_group(b):
    if b < 25:  return "Normal"
    if b < 30:  return "Overweight"      # 25-29.9
    return "Obese"                       # >=30


def ga_group(g):
    if g < 34:  return "<34"             # preterm
    if g <= 36: return "34-36"           # near-term
    return ">=37"                        # full-term


AGE_ORDER = ["<21", "21-30", "31-40"]
BMI_ORDER = ["Normal", "Overweight", "Obese"]
GA_ORDER = ["<34", "34-36", ">=37"]
AXES = {"Age": ("age_grp", AGE_ORDER),
        "BMI": ("bmi_grp", BMI_ORDER),
        "Gestation": ("ga_grp", GA_ORDER)}

DEMO_COLS = ["Age", "Gestational_age", "BMI"]
TARGET = "Heart_Rate"
GROUP = "Subject"

PER_SUBJECT = 220  


def load_full(path=DATA_PATH):
    """Read the full feature CSV and attach subgroup labels."""
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ([GROUP, TARGET] + DEMO_COLS)]
    assert len(feat_cols) == 212, f"expected 212 features, got {len(feat_cols)}"
    df["age_grp"] = df["Age"].apply(age_group)
    df["bmi_grp"] = df["BMI"].apply(bmi_group)
    df["ga_grp"] = df["Gestational_age"].apply(ga_group)
    return df, feat_cols


def build_working_subset(df, per_subject=PER_SUBJECT, seed=SEED):
    parts = []
    for _, sub in df.groupby(GROUP):
        n = min(per_subject, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def get_working_data(per_subject=PER_SUBJECT, path=DATA_PATH):
    """Build the working subset + feature-column list entirely in memory.
    Deterministic (fixed seed). Returns (work_dataframe, feat_cols_list)."""
    df, feat_cols = load_full(path)
    work = build_working_subset(df, per_subject)
    return work, feat_cols


def subject_summary(df):
    return (df.groupby(GROUP)
              .agg(n=(TARGET, "size"), Age=("Age", "first"),
                   GA=("Gestational_age", "first"), BMI=("BMI", "first"),
                   HR=(TARGET, "first"), age_grp=("age_grp", "first"),
                   bmi_grp=("bmi_grp", "first"), ga_grp=("ga_grp", "first")))


if __name__ == "__main__":
    df, feat_cols = load_full()
    print("FULL dataset:", df.shape, "| subjects:", df[GROUP].nunique())
    work = build_working_subset(df)
    print("WORKING subset:", work.shape,
          "| rows/subject:", work.groupby(GROUP).size().min(),
          "-", work.groupby(GROUP).size().max())
