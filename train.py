import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Đọc dữ liệu
du_lieu = pd.read_csv("data.csv")

# 2. Xử lý dữ liệu
du_lieu = du_lieu.drop(columns=["id", "Unnamed: 32"])
du_lieu["diagnosis"] = du_lieu["diagnosis"].map({"M": 0, "B": 1})

# 3. Chọn đặc trưng
dac_trung_quan_trong = [
    "concave points_mean",
    "concave points_worst",
    "area_worst",
    "concavity_mean",
    "radius_worst"
]

X = du_lieu[dac_trung_quan_trong]
y = du_lieu["diagnosis"]

# 4. Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train mô hình
mo_hinh = RandomForestClassifier(
    n_estimators=100,
    max_features="sqrt",
    random_state=42
)
mo_hinh.fit(X_train, y_train)

# 6. Lưu mô hình
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(mo_hinh, f)

print("✅ Mô hình đã được lưu tại models/rf_model.pkl")
