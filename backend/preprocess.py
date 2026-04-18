import os
import pandas as pd

# ===========================
# 1. SET THIS TO YOUR PATH
# ===========================
# Example: r"C:\Users\Shashi\datasets\mimic-iv-3.1"
MIMIC_IV_ROOT = r"/home/shashidhar/datasets/mimic-iv-3.1"  # <-- CHANGE THIS

# ===========================
# CONFIG
# ===========================
N_ADMISSIONS = 3000          # how many admissions to use for prototype
LAB_NROWS = 500000           # how many rows of labevents to load
CHART_NROWS = 500000         # how many rows of chartevents to load

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_CASES = os.path.join(DATA_DIR, "cases.csv")


def check_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def main():
    hosp_dir = os.path.join(MIMIC_IV_ROOT, "hosp")
    icu_dir = os.path.join(MIMIC_IV_ROOT, "icu")

    adm_path = check_path(os.path.join(hosp_dir, "admissions.csv.gz"))
    dx_path = check_path(os.path.join(hosp_dir, "diagnoses_icd.csv.gz"))
    lab_path = check_path(os.path.join(hosp_dir, "labevents.csv.gz"))
    chart_path = check_path(os.path.join(icu_dir, "chartevents.csv.gz"))

    print("[preprocess] Loading admissions...")
    adm = pd.read_csv(adm_path, compression="gzip", low_memory=False)
    adm = adm.sort_values("admittime")
    adm = adm.head(N_ADMISSIONS)
    hadm_ids = set(adm["hadm_id"].tolist())

    print("[preprocess] Loading diagnoses_icd...")
    dx = pd.read_csv(dx_path, compression="gzip", low_memory=False)
    dx = dx[dx["hadm_id"].isin(hadm_ids)]

    # simple diagnosis string per admission
    dx["icd_code"] = dx["icd_code"].astype(str)
    dx_group = (
        dx.groupby("hadm_id")["icd_code"]
        .apply(lambda codes: ", ".join(list(codes)[:5]))
        .reset_index()
        .rename(columns={"icd_code": "dx_codes"})
    )

    print(f"[preprocess] Loading first {LAB_NROWS} rows of labevents...")
    labs = pd.read_csv(lab_path, compression="gzip", nrows=LAB_NROWS, low_memory=False)
    labs = labs[labs["hadm_id"].isin(hadm_ids)]
    labs = labs[["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"]]

    # 🔥 CRITICAL FIX: convert charttime to datetime
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")

    labs["valuenum"] = labs["valuenum"].fillna(0)

    print(f"[preprocess] Loading first {CHART_NROWS} rows of chartevents...")
    charts = pd.read_csv(chart_path, compression="gzip", nrows=CHART_NROWS, low_memory=False)
    charts = charts[charts["hadm_id"].isin(hadm_ids)]
    charts = charts[["subject_id", "hadm_id", "charttime", "itemid", "valuenum", "valueuom"]]

    # 🔥 CRITICAL FIX: convert charttime to datetime
    charts["charttime"] = pd.to_datetime(charts["charttime"], errors="coerce")

    charts["valuenum"] = charts["valuenum"].fillna(0)

    print("[preprocess] Merging admissions and diagnoses...")
    adm = adm.merge(dx_group, on="hadm_id", how="left")

    cases = []
    total = len(adm)

    for idx, row in adm.iterrows():
        hadm_id = row["hadm_id"]
        subject_id = row["subject_id"]
        adm_time = pd.to_datetime(row["admittime"])
        disch_time = pd.to_datetime(row["dischtime"])
        adm_type = str(row.get("admission_type", "UNKNOWN"))
        dx_codes = str(row.get("dx_codes", ""))

        # 24h window from admission
        window_end = adm_time + pd.Timedelta(hours=24)

        # filter labs and charts for this admission + window
        labs_slice = labs[
            (labs["hadm_id"] == hadm_id)
            & (labs["charttime"] >= adm_time)
            & (labs["charttime"] <= window_end)
        ].sort_values("charttime").head(40)

        charts_slice = charts[
            (charts["hadm_id"] == hadm_id)
            & (charts["charttime"] >= adm_time)
            & (charts["charttime"] <= window_end)
        ].sort_values("charttime").head(40)

        labs_text = " | ".join(
            f"{r.itemid}={r.valuenum}{'' if pd.isna(r.valueuom) else r.valueuom}"
            for _, r in labs_slice.iterrows()
        )
        vitals_text = " | ".join(
            f"{r.itemid}={r.valuenum}{'' if pd.isna(r.valueuom) else r.valueuom}"
            for _, r in charts_slice.iterrows()
        )

        pseudo_note = (
            f"Admission type: {adm_type}. "
            f"Primary diagnoses (ICD codes): {dx_codes}. "
            f"Stay from {adm_time} to {disch_time}."
        )

        case_text = (
            f"NOTE: {pseudo_note}\n"
            f"VITALS: {vitals_text}\n"
            f"LABS: {labs_text}\n"
            f"SUBJECT_ID:{subject_id}\n"
            f"HADM_ID:{hadm_id}"
        )

        cases.append(
            {
                "case_id": f"{subject_id}_{hadm_id}",
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "admittime": adm_time,
                "case_text": case_text,
            }
        )

        if (idx + 1) % 500 == 0:
            print(f"[preprocess] Built {idx+1}/{total} cases")

    df_cases = pd.DataFrame(cases)
    df_cases.to_csv(OUT_CASES, index=False)
    print(f"[preprocess] Wrote {len(df_cases)} cases to {OUT_CASES}")


if __name__ == "__main__":
    main()
