import argparse
import json
import os
import re
import tempfile
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CASES_CSV = os.path.join(DATA_DIR, "cases.csv")
DEFAULT_EMBEDDINGS = os.path.join(DATA_DIR, "case_embeddings.npy")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "evaluation")

ICD_PATTERN = re.compile(r"Primary diagnoses \(ICD codes\):\s*([^\.\n]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality with a diagnosis confusion matrix."
    )
    parser.add_argument("--cases-csv", default=DEFAULT_CASES_CSV)
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of nearest neighbors to vote over, excluding self.",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=15,
        help="Show the N most common true labels explicitly in the confusion matrix.",
    )
    return parser.parse_args()


def extract_primary_label(case_text: str) -> str:
    match = ICD_PATTERN.search(str(case_text))
    if not match:
        return "UNKNOWN"

    codes = [code.strip() for code in match.group(1).split(",") if code.strip()]
    return codes[0] if codes else "UNKNOWN"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def choose_prediction(neighbor_labels):
    counts = Counter(neighbor_labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_predictions(embeddings: np.ndarray, labels: list[str], k: int):
    normalized = normalize_rows(embeddings.astype(np.float32))
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)

    top_idx = np.argpartition(similarity, -k, axis=1)[:, -k:]
    row_ids = np.arange(similarity.shape[0])[:, None]
    ordered = np.argsort(similarity[row_ids, top_idx], axis=1)[:, ::-1]
    neighbor_idx = top_idx[row_ids, ordered]

    predictions = []
    neighbor_labels_all = []
    for indices in neighbor_idx:
        n_labels = [labels[idx] for idx in indices]
        neighbor_labels_all.append(n_labels)
        predictions.append(choose_prediction(n_labels))

    return predictions, neighbor_idx, neighbor_labels_all


def collapse_label(label: str, kept_labels: set[str]) -> str:
    return label if label in kept_labels else "OTHER"


def save_heatmap(cm: np.ndarray, labels: list[str], output_path: str):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix for Retrieval-Based Diagnosis Prediction")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > threshold else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_normalized_heatmap(cm: np.ndarray, labels: list[str], output_path: str):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
    import matplotlib.pyplot as plt

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized = cm / row_sums

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix")

    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            value = normalized[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cases_df = pd.read_csv(args.cases_csv)
    embeddings = np.load(args.embeddings)

    if len(cases_df) != len(embeddings):
        raise ValueError(
            f"cases ({len(cases_df)}) and embeddings ({len(embeddings)}) must match"
        )

    if args.k < 1:
        raise ValueError("--k must be at least 1")
    if args.k >= len(cases_df):
        raise ValueError("--k must be smaller than the number of cases")

    cases_df["true_label"] = cases_df["case_text"].map(extract_primary_label)
    labels = cases_df["true_label"].tolist()

    predictions, neighbor_idx, neighbor_labels_all = build_predictions(
        embeddings=embeddings,
        labels=labels,
        k=args.k,
    )
    cases_df["pred_label"] = predictions
    cases_df["neighbor_indices"] = ["|".join(map(str, row)) for row in neighbor_idx]
    cases_df["neighbor_labels"] = ["|".join(row) for row in neighbor_labels_all]

    top_true_labels = [
        label
        for label, _ in Counter(labels).most_common(args.top_labels)
    ]
    kept_labels = set(top_true_labels)
    matrix_labels = top_true_labels + ["OTHER"]

    cases_df["true_label_grouped"] = cases_df["true_label"].map(
        lambda label: collapse_label(label, kept_labels)
    )
    cases_df["pred_label_grouped"] = cases_df["pred_label"].map(
        lambda label: collapse_label(label, kept_labels)
    )

    cm = confusion_matrix(
        cases_df["true_label_grouped"],
        cases_df["pred_label_grouped"],
        labels=matrix_labels,
    )

    accuracy = accuracy_score(cases_df["true_label"], cases_df["pred_label"])
    grouped_accuracy = accuracy_score(
        cases_df["true_label_grouped"], cases_df["pred_label_grouped"]
    )

    metrics = {
        "num_cases": int(len(cases_df)),
        "k": int(args.k),
        "top_labels_visualized": int(args.top_labels),
        "exact_match_accuracy": float(accuracy),
        "grouped_accuracy": float(grouped_accuracy),
        "num_unique_true_labels": int(cases_df["true_label"].nunique()),
    }

    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    cm_csv_path = os.path.join(args.output_dir, "confusion_matrix.csv")
    cm_png_path = os.path.join(args.output_dir, "confusion_matrix.png")
    cm_normalized_png_path = os.path.join(
        args.output_dir, "confusion_matrix_normalized.png"
    )
    metrics_path = os.path.join(args.output_dir, "metrics.json")

    cases_df[
        [
            "case_id",
            "subject_id",
            "hadm_id",
            "true_label",
            "pred_label",
            "true_label_grouped",
            "pred_label_grouped",
            "neighbor_indices",
            "neighbor_labels",
        ]
    ].to_csv(predictions_path, index=False)

    pd.DataFrame(cm, index=matrix_labels, columns=matrix_labels).to_csv(cm_csv_path)

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_heatmap(cm, matrix_labels, cm_png_path)
    save_normalized_heatmap(cm, matrix_labels, cm_normalized_png_path)

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved confusion matrix CSV to {cm_csv_path}")
    print(f"Saved confusion matrix image to {cm_png_path}")
    print(f"Saved normalized confusion matrix image to {cm_normalized_png_path}")


if __name__ == "__main__":
    main()
