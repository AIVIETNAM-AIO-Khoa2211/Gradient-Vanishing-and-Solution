# Cấu trúc code và phân công

Đề tài: nghiên cứu các giải pháp cho hiện tượng vanishing / exploding gradient trên mạng NN viết từ đầu bằng NumPy.

## 1. Cấu trúc thư mục

```
project/
├── src/
│   ├── __init__.py          # (hai dấu gạch dưới mỗi bên)
│   ├── activation.py        # hàm kích hoạt + đạo hàm
│   ├── initializer.py       # WeightInitializer, khởi tạo W, b
│   ├── neural_network.py    # class NeuralNetwork (forward / backward / fit)
│   └── visualize.py         # vẽ biểu đồ gradient, chỉ số vanish / explode
├── tests/                   # test đạo hàm, init (mỗi người test file của mình)
├── results/                 # history .npz, hình xuất ra
├── Main.ipynb               # thực nghiệm, đọc kết quả, so sánh
├── requirements.txt         # numpy, matplotlib, scikit-learn
├── .gitignore
└── README.md
```

## 2. Phân công

| File / việc | Phụ trách | Ghi chú |
|---|---|---|
| `src/activation.py` | Khải | Hàm kích hoạt + đạo hàm |
| `src/initializer.py` | Triều Vỹ | `WeightInitializer`, khởi tạo W, b |
| `src/neural_network.py` | Duy | Import từ `activation.py` và `initializer.py`, không định nghĩa lại, nếu 2 file trên vẫn chưa xong thì có thể gọi thư viện |
| `src/visualize.py` | Minh Khoa | Đã có bản nháp, cần một người rà soát và chạy thử |
| Phần giới thiệu report | Huy | Không phụ thuộc vào code |

## 3. Thứ tự phụ thuộc

1. `activation.py` và `initializer.py` làm song song, độc lập nhau.
2. `neural_network.py` chỉ cần hai file trên đúng "hợp đồng" là ghép được.
3. `visualize.py` chỉ đọc `history` do `fit` trả về, không phụ thuộc trực tiếp vào các file kia.
4. Baseline trong `Main.ipynb` chạy được ngay khi có đủ 1–2.