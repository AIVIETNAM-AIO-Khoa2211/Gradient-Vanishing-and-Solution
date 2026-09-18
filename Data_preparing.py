import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Xác định đường dẫn thư mục data nằm CÙNG CẤP với file Python này
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Tạo thư mục data nếu chưa tồn tại
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Tải bộ dữ liệu MNIST từ internet
print("Đang tải dữ liệu MNIST, vui lòng đợi...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

# 3. Chuẩn hóa giá trị điểm ảnh về [0, 1]
X = X / 255.0

# 4. Chia dữ liệu: Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 5. Lưu các tập dữ liệu vào DATA_DIR vừa xác định ở trên
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)

np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)

np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

# 6. In kết quả kiểm tra
print("\n--- HOÀN THÀNH ---")
print("Đường dẫn lưu data:", DATA_DIR)
print("Kích thước tập Train:", X_train.shape)
print("Kích thước tập Val:  ", X_val.shape)
print("Kích thước tập Test: ", X_test.shape)