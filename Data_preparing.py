import os
import numpy as np
import matplotlib.pyplot as plt
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
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)


# 5. Lưu các tập dữ liệu vào DATA_DIR vừa xác định ở trên
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)

np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)

# 6. In kết quả kiểm tra
print("\n--- HOÀN THÀNH ---")
print("Đường dẫn lưu data:", DATA_DIR)
print("Kích thước tập Train:", X_train.shape)
print("Kích thước tập Val:  ", X_val.shape)

# 7. Trực quan hóa dữ liệu
# --- Thống kê phân bố nhãn ---
counts = np.bincount(y, minlength=10)
percents = counts / counts.sum() * 100
labels = np.arange(10)

# --- Vẽ biểu đồ ---
fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(labels, counts, color="tab:blue", edgecolor="black", linewidth=0.5)

for bar, c, p in zip(bars, counts, percents):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{c:,}\n({p:.2f}%)",
        ha="center", va="bottom", fontsize=8,
    )

ax.axhline(counts.mean(), color="red", linestyle="--", linewidth=1,
           label=f"Trung bình ({counts.mean():,.0f} mẫu/lớp)")
ax.set_xticks(labels)
ax.set_xlabel("Nhãn")
ax.set_ylabel("Số mẫu")
ax.set_title("Phân bố nhãn trên bộ dữ liệu MNIST")
ax.set_ylim(0, counts.max() * 1.15)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

# --- In bảng LaTeX từ số liệu thật ---
fmt_int = lambda n: f"{n:,}".replace(",", r"\,")
fmt_pct = lambda p: f"{p:.2f}".replace(".", ",")

print(r"\textbf{Số mẫu} & " + " & ".join(fmt_int(c) for c in counts) + r" \\")
print(r"\textbf{Tỷ lệ (\%)} & " + " & ".join(fmt_pct(p) for p in percents) + r" \\")
