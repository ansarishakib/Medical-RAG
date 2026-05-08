"""
preprocessing.py — Clean & normalise structured_output01.csv
Run: python preprocessing.py
Output: data/cleaned_prescriptions.csv
"""

import re
import pandas as pd

INPUT_PATH  = "data/structured_output01.csv"
OUTPUT_PATH = "data/cleaned_prescriptions.csv"


# ────────────────────────────────────────────────────────────
# 1.  LOAD
# ────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded  {len(df)} rows × {len(df.columns)} columns")
    return df


# ────────────────────────────────────────────────────────────
# 2.  DROP USELESS COLUMNS
# ────────────────────────────────────────────────────────────

def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ocr_quality_score is 100 % null — remove it entirely."""
    before = set(df.columns)
    df = df.dropna(axis=1, how="all")
    dropped = before - set(df.columns)
    if dropped:
        print(f"[DROP COLUMNS]  {dropped}")
    return df


# ────────────────────────────────────────────────────────────
# 3.  STRIP WHITESPACE FROM ALL STRING COLUMNS
# ────────────────────────────────────────────────────────────

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# ────────────────────────────────────────────────────────────
# 4.  REPLACE EMBEDDED NEWLINES  (\n → " | ")
# ────────────────────────────────────────────────────────────

def fix_newlines(df: pd.DataFrame) -> pd.DataFrame:
    """
    medicines and dosage_intake can contain literal \\n separating
    multiple drugs / doses.  Replace with ' | ' for readability.
    """
    for col in ["medicines", "dosage_intake", "patient_type"]:
        if col in df.columns:
            df[col] = df[col].str.replace(r"\\n|\n", " | ", regex=True)
            # collapse any double spaces left behind
            df[col] = df[col].str.replace(r"  +", " ", regex=True).str.strip()
    return df


# ────────────────────────────────────────────────────────────
# 5.  FIX patient_type TYPOS & MERGE RARE CATEGORIES
# ────────────────────────────────────────────────────────────

PATIENT_TYPE_MAP = {
    "Cardiovescular":               "Cardiovascular",
    "Gastrointestinal | Heptobiliary": "Gastrointestinal/Hepatobiliary",
    # rare / ambiguous categories → broader group
    "Symptomatic":                  "General",
    "Nephrological":                "Renal",
}

def fix_patient_type(df: pd.DataFrame) -> pd.DataFrame:
    df["patient_type"] = df["patient_type"].replace(PATIENT_TYPE_MAP)
    return df


# ────────────────────────────────────────────────────────────
# 6.  FIX DIAGNOSIS TEXT
#     - trailing/leading whitespace (already done in step 3)
#     - known spelling mistakes
#     - title-case for lower-case entries
# ────────────────────────────────────────────────────────────

DIAGNOSIS_MAP = {
    "Ranal Calculi":                        "Renal Calculi",
    "Acute ST-elevation myocardial infraction": "Acute ST-elevation myocardial infarction",
    "COVID-19 prophulaxis":                 "COVID-19 prophylaxis",
    "Sciatica due to prolapsed intervertabral disc":
        "Sciatica due to prolapsed intervertebral disc",
    "GRED with hypothyroidism ,migraine ,IBS, Asthama":
        "GERD with hypothyroidism, migraine, IBS, Asthma",
    "Estogen deficiency":                   "Estrogen deficiency",
    "Severe invasion fungal infection":     "Severe invasive fungal infection",
    "cardiac condition":                    "Cardiac condition",
}

def fix_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    df["diagnosis"] = df["diagnosis"].replace(DIAGNOSIS_MAP)
    # Sentence-case everything (keep existing capitalisation for abbreviations)
    def sentence_case(val):
        if pd.isna(val):
            return val
        # only lower-case the first char if the whole string is lowercase
        if val[0].islower():
            return val[0].upper() + val[1:]
        return val
    df["diagnosis"] = df["diagnosis"].apply(sentence_case)
    return df


# ────────────────────────────────────────────────────────────
# 7.  FIX MEDICINE TYPOS
# ────────────────────────────────────────────────────────────

MEDICINE_FIXES = {
    # OCR prefix errors
    r"\bTeb\b":         "Tab",
    # Drug name misspellings
    r"\bAmoxixillin\b": "Amoxicillin",
    r"\bAmoxicilin\b":  "Amoxicillin",
    r"\bFexofenadiac\b":"Fexofenadine",
    r"\bHumilin\b":     "Humulin",
    r"\bAmisome\b":     "Ambisome",
}

def fix_medicines(df: pd.DataFrame) -> pd.DataFrame:
    def apply_fixes(val):
        if pd.isna(val):
            return val
        for pattern, replacement in MEDICINE_FIXES.items():
            val = re.sub(pattern, replacement, val)
        return val
    df["medicines"] = df["medicines"].apply(apply_fixes)
    return df


# ────────────────────────────────────────────────────────────
# 8.  NORMALISE DOSAGE FREQUENCY WORDING
#     Standardise to clinical shorthand: OD BD TDS QID
# ────────────────────────────────────────────────────────────

FREQ_MAP = {
    # once a day
    r"1 time (a|per) day":   "OD (once daily)",
    r"1 time daily":         "OD (once daily)",
    r"once( a day)?":        "OD (once daily)",
    # twice a day
    r"2 times (a|per) day":  "BD (twice daily)",
    r"2 times daily":        "BD (twice daily)",
    r"twice (a day|daily)":  "BD (twice daily)",
    # three times a day
    r"3 times (a|per) day":  "TDS (three times daily)",
    r"3 times daily":        "TDS (three times daily)",
    # four times a day
    r"4 times (a|per) day":  "QID (four times daily)",
    r"4 times daily":        "QID (four times daily)",
    # ten times a day (e.g. eye drops)
    r"10 times (a|per) day": "10 times daily",
}

def normalise_dosage(df: pd.DataFrame) -> pd.DataFrame:
    def apply_freq(val):
        if pd.isna(val):
            return val
        for pattern, replacement in FREQ_MAP.items():
            val = re.sub(pattern, replacement, val, flags=re.IGNORECASE)
        return val
    df["dosage_intake"] = df["dosage_intake"].apply(apply_freq)
    return df


# ────────────────────────────────────────────────────────────
# 9.  FILL / HANDLE MISSING VALUES
# ────────────────────────────────────────────────────────────

def fill_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy per column:
      - High null %  (doctor, hospital, patient_name, gender, age)
        → fill with "Unknown" / 0 so downstream code never breaks
      - medicines / dosage_intake
        → fill "Not specified" (required for RAG document)
      - diagnosis
        → fill "Unspecified" (only 5 missing)
    """
    fill_str = {
        "patient_name":  "Unknown",
        "gender":        "Unknown",
        "doctor":        "Unknown",
        "hospital":      "Unknown",
        "medicines":     "Not specified",
        "dosage_intake": "Not specified",
        "diagnosis":     "Unspecified",
    }
    for col, val in fill_str.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    if "age_in_years" in df.columns:
        df["age_in_years"] = df["age_in_years"].fillna(0).astype(int)

    return df


# ────────────────────────────────────────────────────────────
# 10. ADD HELPER COLUMN: age_group
# ────────────────────────────────────────────────────────────

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    def categorise(age):
        if age == 0:    return "Unknown"
        if age < 13:    return "Child"
        if age < 18:    return "Adolescent"
        if age < 60:    return "Adult"
        return          "Senior"
    df["age_group"] = df["age_in_years"].apply(categorise)
    return df


# ────────────────────────────────────────────────────────────
# PIPELINE
# ────────────────────────────────────────────────────────────

def preprocess(input_path: str = INPUT_PATH,
               output_path: str = OUTPUT_PATH) -> pd.DataFrame:

    df = load(input_path)

    steps = [
        ("Drop empty columns",       drop_empty_columns),
        ("Strip whitespace",         strip_whitespace),
        ("Fix embedded newlines",    fix_newlines),
        ("Fix patient_type",         fix_patient_type),
        ("Fix diagnosis text",       fix_diagnosis),
        ("Fix medicine typos",       fix_medicines),
        ("Normalise dosage wording", normalise_dosage),
        ("Fill missing values",      fill_nulls),
        ("Add age_group column",     add_age_group),
    ]

    for name, fn in steps:
        df = fn(df)
        print(f"  ✓  {name}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  ({len(df)} rows × {len(df.columns)} columns)")

    # Quick summary
    print("\n── Column null counts after cleaning ──")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "  None — fully clean!")

    print("\n── patient_type distribution ──")
    print(df["patient_type"].value_counts().to_string())

    return df


if __name__ == "__main__":
    preprocess()
