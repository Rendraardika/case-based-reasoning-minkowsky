from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

DATASET_PATH = "dataset.handphone.csv"

# -------------------------------------------------
# Normalisasi input user
# -------------------------------------------------
def normalize_number(text):
    text = text.lower().replace(",", ".").strip()

    if "juta" in text or "jt" in text or "m" in text:
        num = float(re.sub(r"[^0-9.]", "", text))
        return num * 1_000_000

    num = re.sub(r"[^0-9.]", "", text)
    return float(num)

# -------------------------------------------------
# Cleaning angka dataset
# -------------------------------------------------
def clean_number(x):
    try:
        return float(re.sub(r"[^0-9.]", "", str(x)))
    except:
        return 0.0

# -------------------------------------------------
# Minkowski Distance
# -------------------------------------------------
def minkowski(a, b, p=2):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.power(np.sum(np.abs(a - b)**p), 1/p)

# -------------------------------------------------
# Similarity 0–1
# -------------------------------------------------
def similarity_score(distance, max_distance=100000):
    score = 1 - (distance / max_distance)
    return max(0.0, min(1.0, score))

# -------------------------------------------------
# CBR TOP 3 RECOMMENDATION
# -------------------------------------------------
def cbr_predict_top3(input_case, p=2):
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = ["Harga", "Ram", "Memori_internal", "Ukuran_layar", "Kapasitas_baterai"]

    input_values = [input_case[col] for col in numeric_cols]

    hasil = []
    for _, row in df.iterrows():
        dist = minkowski(np.array(input_values), row[numeric_cols].values, p)
        sim = similarity_score(dist)

        hasil.append({
            "data": row,
            "distance": float(dist),
            "similarity": float(sim)
        })

    hasil_sorted = sorted(hasil, key=lambda x: x["distance"])
    return hasil_sorted[:3]

# -------------------------------------------------
# Perhitungan akurasi 70:30
# -------------------------------------------------
def compute_metrics(test_ratio=0.3, p=2):
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = ["Harga", "Ram", "Memori_internal", "Ukuran_layar", "Kapasitas_baterai"]

    for col in numeric_cols:
        df[col] = df[col].apply(clean_number)

    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_train = int((1 - test_ratio) * n_total)

    train_df = df_shuffled.iloc[:n_train]
    test_df = df_shuffled.iloc[n_train:]

    y_true = []
    y_pred = []

    for _, test_row in test_df.iterrows():
        test_case = test_row[numeric_cols].values

        min_dist = float("inf")
        best_match = None

        for _, train_row in train_df.iterrows():
            dist = minkowski(test_case, train_row[numeric_cols].values, p)
            if dist < min_dist:
                min_dist = dist
                best_match = train_row

        y_true.append(test_row["Brand"])
        y_pred.append(best_match["Brand"])

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "train_n": len(train_df),
        "test_n": len(test_df)
    }

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hasil", methods=["POST"])
def hasil():
    try:
        input_case = {
            "Harga": normalize_number(request.form["harga"]),
            "Ram": normalize_number(request.form["ram"]),
            "Memori_internal": normalize_number(request.form["memori"]),
            "Ukuran_layar": normalize_number(request.form["layar"]),
            "Kapasitas_baterai": normalize_number(request.form["baterai"]),
        }

        rekomendasi = cbr_predict_top3(input_case)

        sim_top = rekomendasi[0]["similarity"]

        if sim_top >= 0.8:
            kesimpulan = "Sangat Mirip (Rekomendasi paling relevan)"
        elif sim_top >= 0.5:
            kesimpulan = "Mirip (Perbedaan spesifikasi tidak terlalu jauh)"
        elif sim_top >= 0.2:
            kesimpulan = "Cukup Mirip (Masih bisa dipertimbangkan)"
        else:
            kesimpulan = "Tidak Mirip (Perbedaan spesifikasi terlalu besar)"

        metrics = compute_metrics()

        return render_template(
            "hasil.html",
            rekomendasi=rekomendasi,
            kesimpulan=kesimpulan,
            metrics=metrics
        )

    except Exception as e:
        return f"Terjadi error: {e}"

@app.route("/akurasi")
def akurasi():
    metrics = compute_metrics()
    return render_template("akurasi.html", M=metrics)

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
