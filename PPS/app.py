"""Máy chủ Flask: nhận file xlsx, chạy thuật toán trong numerics.REGISTRY, trả JSON."""
import io, uuid
from collections import OrderedDict
import numpy as np
from flask import Flask, jsonify, render_template, request
from openpyxl import load_workbook
from numerics import REGISTRY, clean

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
STORE = OrderedDict()   # token -> dict(x, y, name); giữ tối đa 20 bộ dữ liệu trong bộ nhớ

SAMPLES = {
    "runge": ("1/(25x²+1) trên [−1, 1]", np.linspace(-1, 1, 2001), lambda x: 1 / (25 * x ** 2 + 1)),
    "exp3": ("3^x trên [−2, 2]", np.linspace(-2, 2, 2001), lambda x: 3.0 ** x),
    "slide": ("Slide Newton (6 điểm)", np.array([1.2, 1.3, 1.5, 1.7, 1.8, 2.0]), lambda x: np.array([3.172, 3.695, 3.974, 4.152, 3.769, 3.112])),
    "cau1": ("Câu 1 Newton lùi (4 điểm)", np.array([-2.5, -2.0, -1.5, -1.0]), lambda x: np.array([15.93, 16.17, 15.53, 15.13])),
    "inverse": ("Ví dụ nội suy ngược (9 điểm)", np.round(np.arange(1.2, 2.81, 0.2), 1), lambda x: np.array([4.02, 3.98, 4.23, 3.67, 2.99, 1.24, 0.87, 2.08, 4.54])),
    "spline": ("Spline (7 điểm, đọc gần đúng từ hình slide)", np.array([0.3, 0.7, 1.2, 1.5, 1.8, 2.2, 2.6]), lambda x: np.array([1.2, 2.7, 3.0, 2.3, 3.2, 1.0, 0.8])),
    "lsq": ("Bình phương tối thiểu: y = a·e^(bx) − 2", np.round(np.arange(1.0, 2.01, 0.1), 1), lambda x: np.array([-0.641, -0.498, -0.34, -0.165, 0.028, 0.241, 0.477, 0.737, 1.025, 1.343, 1.695])),
    "sin": ("sin(x) trên [0, 10]", np.linspace(0, 10, 2001), np.sin),
}


def to_float(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))   # chấp nhận "1,1" kiểu Việt Nam
    except ValueError:
        return None


def parse_xlsx(stream):
    ws = load_workbook(stream, read_only=True, data_only=True).active
    pts, skipped = [], 0
    for row in ws.iter_rows(values_only=True):
        a, b = (list(row[:2]) + [None, None])[:2]
        a, b = to_float(a), to_float(b)
        if a is None or b is None:
            skipped += 1
        else:
            pts.append((a, b))
    if len(pts) < 3:
        raise ValueError("File cần ít nhất 3 dòng số: cột A là x, cột B là f(x).")
    pts.sort()
    x, y = np.array(pts, float).T
    if np.any(np.diff(x) == 0):
        raise ValueError("Cột x có giá trị trùng nhau; các mốc nội suy phải phân biệt.")
    return x, y, skipped


def store(x, y, name, skipped=0):
    token = uuid.uuid4().hex
    STORE[token] = dict(x=x, y=y, name=name)
    while len(STORE) > 20:
        STORE.popitem(last=False)
    d = np.diff(x)
    s = max(1, len(x) // 500)
    return dict(token=token, name=name, count=len(x), xmin=x[0], xmax=x[-1], ymin=y.min(), ymax=y.max(),
                step=float(np.median(d)), uniform=bool(np.allclose(d, d[0], rtol=1e-6, atol=0)), skipped=skipped,
                preview=dict(x=x[::s], y=y[::s]))


@app.errorhandler(Exception)
def on_error(e):
    if hasattr(e, "code") and hasattr(e, "description"):      # lỗi HTTP của Flask
        return jsonify(error=e.description), e.code
    return jsonify(error=str(e)), 400 if isinstance(e, (ValueError, KeyError, np.linalg.LinAlgError)) else 500


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/algorithms")
def algorithms():
    return jsonify([{k: v for k, v in a.items() if k != "fn"} for a in REGISTRY.values()])


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith((".xlsx", ".xlsm")):
        raise ValueError("Hãy chọn file .xlsx (2 cột: x, f(x)).")
    x, y, skipped = parse_xlsx(io.BytesIO(f.read()))
    return jsonify(clean(store(x, y, f.filename, skipped)))


@app.post("/api/sample/<name>")
def sample(name):
    label, x, fn = SAMPLES[name]
    return jsonify(clean(store(x, fn(x), "Mẫu: " + label)))


@app.post("/api/run")
def run():
    req = request.get_json(force=True)
    ds = STORE.get(req.get("token"))
    if ds is None:
        raise ValueError("Chưa có dữ liệu (hoặc máy chủ đã khởi động lại). Hãy tải file lên lại.")
    algo = REGISTRY[req["algorithm"]]
    a, b = to_float(req.get("a")), to_float(req.get("b"))
    if a is None or b is None or not a < b:
        raise ValueError("Cần nhập a < b (số thực).")
    if a < ds["x"][0] or b > ds["x"][-1]:
        raise ValueError(f"[a, b] phải nằm trong miền dữ liệu [{ds['x'][0]:.6g}, {ds['x'][-1]:.6g}].")
    params = {}
    for prm in algo["params"]:
        v = req.get("params", {}).get(prm["name"], prm["default"])
        if prm["type"] == "int":
            params[prm["name"]] = int(float(v))
        elif prm["type"] == "number":
            params[prm["name"]] = to_float(v)
        elif prm["type"] == "checkbox":
            params[prm["name"]] = bool(v)
        else:
            params[prm["name"]] = str(v)
    out = algo["fn"](ds["x"], ds["y"], a, b, params)
    out.pop("plots", None)          # giao diện không hiển thị đồ thị: chỉ liệt kê các bước và kết quả
    return jsonify(clean(dict(out, title=algo["name"])))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
