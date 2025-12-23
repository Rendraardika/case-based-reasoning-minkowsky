from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re

app = Flask(__name__)

DATASET_PATH = "dataset.handphone.csv"


#  VALIDASI WAJIB DIISI

def require_input(value, field_name):
    if value.strip() == "":
        raise ValueError(f"Input '{field_name}' wajib diisi.")
    return value


# NORMALISASI ANGKA UMUM

def normalize_number(text):
    text = text.lower().strip()

    # harga jutaan
    if "juta" in text or "jt" in text:
        text = text.replace(",", ".")
        num = re.sub(r"[^0-9.]", "", text)
        return float(num) * 1_000_000

    # harga memakai "m"
    if re.search(r"\d+\s*m\b", text):
        text = text.replace(",", ".")
        num = re.sub(r"[^0-9.]", "", text)
        return float(num) * 1_000_000

    # angka umum
    text = text.replace(",", ".")
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group())

    return 0.0


# NORMALISASI RESOLUSI KAMERA 
 

def normalize_camera(text):
    text = text.lower().strip()
    text = text.replace(",", " ").replace("+", " ")

    numbers = re.findall(r"\d+", text)
    if not numbers:
        return 0.0

    return float(max([int(n) for n in numbers]))


# CLEAN DATASET ANGKA

def clean_number(x):
    text = str(x).lower().strip()
    text = text.replace(",", ".")
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return 0.0


def clean_camera(x):
    return normalize_camera(str(x))



# MINKOWSKI DISTANCE

def minkowski(a, b, p=2):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.power(np.sum(np.abs(a - b) ** p), 1 / p)


# SIMILARITY 0–1
 
def similarity_score(distance):
    max_distance = 6 * 1_000_000  # karena ada 6 fitur
    score = 1 - (distance / max_distance)
    return max(0.0, min(1.0, score))



def cbr_predict_top3(input_case, p=2):
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = ["Harga", "Ram", "Memori_internal", "Ukuran_layar",
                    "Kapasitas_baterai", "Resolusi_kamera"]

    # BERSIHKAN DATASET
    for col in numeric_cols:
        if col == "Resolusi_kamera":
            df[col] = df[col].apply(clean_camera)
        else:
            df[col] = df[col].apply(clean_number)

    input_values = [input_case[col] for col in numeric_cols]

    hasil = []
    for _, row in df.iterrows():
        row_values = [row[col] for col in numeric_cols]

        dist = minkowski(input_values, row_values, p)
        sim = similarity_score(dist)

        hasil.append({
            "data": row,
            "distance": float(dist),
            "similarity": float(sim)
        })

    hasil_sorted = sorted(hasil, key=lambda x: x["distance"])
    return hasil_sorted[:3]


# EVALUASI KINERJA CBR
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_cbr_performance(p=2, test_size=0.3, random_state=42):
    df = pd.read_csv(DATASET_PATH)

    numeric_cols = ["Harga", "Ram", "Memori_internal", "Ukuran_layar",
                    "Kapasitas_baterai", "Resolusi_kamera"]

    # Bersihkan data numerik
    for col in numeric_cols:
        if col == "Resolusi_kamera":
            df[col] = df[col].apply(clean_camera)
        else:
            df[col] = df[col].apply(clean_number)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    y_true = []
    y_pred = []

    for _, test_row in test_df.iterrows():
        input_case = {col: test_row[col] for col in numeric_cols}

        hasil = []
        for _, train_row in train_df.iterrows():
            dist = minkowski(
                [input_case[c] for c in numeric_cols],
                [train_row[c] for c in numeric_cols],
                p
            )
            hasil.append((train_row["Brand"], dist))

        # Ambil brand dengan jarak terdekat
        predicted_brand = min(hasil, key=lambda x: x[1])[0]

        y_true.append(test_row["Brand"])
        y_pred.append(predicted_brand)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
    }


# ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/hasil", methods=["POST"])
def hasil():

    try:
        # VALIDASI WAJIB DIISI
        harga = require_input(request.form["harga"], "Harga")
        ram = require_input(request.form["ram"], "RAM")
        mem = require_input(request.form["memori"], "Memori Internal")
        layar = require_input(request.form["layar"], "Ukuran Layar")
        bat = require_input(request.form["baterai"], "Kapasitas Baterai")
        kamera = require_input(request.form["kamera"], "Resolusi Kamera")

        # NORMALISASI INPUT
        input_case = {
            "Harga": normalize_number(harga),
            "Ram": normalize_number(ram),
            "Memori_internal": normalize_number(mem),
            "Ukuran_layar": normalize_number(layar),
            "Kapasitas_baterai": normalize_number(bat),
            "Resolusi_kamera": normalize_camera(kamera),
        }

        rekomendasi = cbr_predict_top3(input_case)
        sim_top = rekomendasi[0]["similarity"]

        nama_hp = [
    rekomendasi[0]["data"]["Nama_hp"],
    rekomendasi[1]["data"]["Nama_hp"],
    rekomendasi[2]["data"]["Nama_hp"]
]

        kesimpulan = (
            f"Berikut adalah 3 rekomendasi HP yang paling sesuai dengan kebutuhan Anda: "
            f"{nama_hp[0]}, {nama_hp[1]}, dan {nama_hp[2]}. "
            f"Urutan ini dihasilkan berdasarkan tingkat kemiripan spesifikasi terhadap data yang Anda masukkan."
        )

        return render_template("hasil.html", rekomendasi=rekomendasi, kesimpulan=kesimpulan)

    except Exception as e:
        return f"Terjadi error: {e}"

@app.route("/evaluasi")
def evaluasi():
    metrics = evaluate_cbr_performance()

    return render_template("evaluasi.html", metrics=metrics)




if __name__ == "__main__":
    app.run(debug=True)
