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

## 3. "Hợp đồng" giữa các file

Ba file gọi lẫn nhau nên mọi người cần dùng đúng chữ ký hàm dưới đây, nếu không lúc ghép sẽ lỗi.

### `activation.py`

```python
get_activation(name: str) -> tuple[callable, callable | None]   # (fn, dfn)
```

- Tên hợp lệ (chữ thường): `"relu"`, `"sigmoid"`, `"tanh"`, `"none"`, `"leaky_relu"`, `"elu"`, `"softmax"`.
- Input `Z` có shape `(n_units, n_samples)`; hàm và đạo hàm trả về array cùng shape, không sửa `Z` tại chỗ.
- Đạo hàm là theo `Z`, tính tại `Z`.
- `softmax` tính theo `axis=0`, ổn định số học, và có `dfn = None` (luôn đi cùng cross-entropy, `dZ = A - Y` đã xử lý trong class).
- Tên không tồn tại thì `raise ValueError` kèm danh sách tên hợp lệ.
- Test bắt buộc: đạo hàm so với sai phân trung tâm (sai số tương đối < 1e-5).

### `initializer.py`

```python
initialize_params(layer_dims: list[int], scheme: str = "uniform", seed: int = 42) -> tuple[dict, dict]   # (W, b)
```

- `W`: dict `{i: array shape (layer_dims[i], layer_dims[i-1])}`, `i = 1..L`, không có key 0.
- `b`: dict `{i: np.zeros((layer_dims[i], 1))}`, cùng key với `W`.
- Scheme: `"uniform"` (baseline: `rng.random(shape) - 0.5`), `"xavier"`, `"he"`, `"lecun"`, tùy chọn `"zeros"`.
- Chỉ tạo **một** `rng` duy nhất từ `seed` và duyệt layer theo thứ tự `1..L`; cùng seed thì kết quả phải giống hệt nhau.
- Scheme không tồn tại thì `raise ValueError`.

### `neural_network.py`

```python
NeuralNetwork(layer_dims, activations=None, init="uniform", seed=42)
```

- `activations` có độ dài `len(layer_dims)`, phần tử 0 là placeholder `"none"`; phần tử cuối bắt buộc là `"softmax"`.
- `fit(...)` trả về `history` (dict các mảng, shape `[số lần ghi, số layer]`) để đưa vào `visualize.py`.
- Import đúng tên file: `from .activation import get_activation` và `from .initializer import initialize_params`.

## 4. Quy ước làm việc

- Mỗi người một nhánh git, chỉ sửa file của mình.
- `Main.ipynb` do một người phụ trách để tránh xung đột; xóa output của notebook trước khi commit.
- Cho `results/**/*.npz`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/` vào `.gitignore`.
- Chạy code từ thư mục gốc `project/`, import theo kiểu `from src.neural_network import NeuralNetwork`.

## 5. Thứ tự phụ thuộc

1. `activation.py` và `initializer.py` làm song song, độc lập nhau.
2. `neural_network.py` chỉ cần hai file trên đúng "hợp đồng" là ghép được.
3. `visualize.py` chỉ đọc `history` do `fit` trả về, không phụ thuộc trực tiếp vào các file kia.
4. Baseline trong `Main.ipynb` chạy được ngay khi có đủ 1–2.