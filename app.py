from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = os.path.join("models", "rf_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Đặc trưng được chọn
dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form
            values = [float(request.form.get(feat)) for feat in dac_trung_quan_trong]
            df = pd.DataFrame([values], columns=dac_trung_quan_trong)

            # Dự đoán
            prediction = model.predict(df)[0]
            if prediction == 0:
                result = "❌ Khối u ÁC TÍNH"
            else:
                result = "✅ Khối u LÀNH TÍNH"
        except:
            result = "⚠️ Vui lòng nhập số hợp lệ!"
    return render_template("index.html", result=result, features=dac_trung_quan_trong)

if __name__ == "__main__":
    app.run(debug=True)
