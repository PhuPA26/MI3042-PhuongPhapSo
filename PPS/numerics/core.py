"""Hàm số học dùng chung cho các thuật toán."""
import re
import numpy as np


def subset(x, y, a, b):
    eps = 1e-12 * max(1.0, abs(a), abs(b))
    m = (x >= a - eps) & (x <= b + eps)
    return x[m], y[m]


def cheb_targets(a, b, n):
    """n+1 nghiệm của T_{n+1}: t_i = cos(pi/(2(n+1)) + i*pi/(n+1)), rồi đổi biến về [a,b]."""
    i = np.arange(n + 1)
    t = np.cos(np.pi / (2 * (n + 1)) + i * np.pi / (n + 1))
    return t, ((b - a) * t + (b + a)) / 2


def cheb_T(k, t):
    """T_k(t) theo công thức truy hồi T_{k+1} = 2t T_k - T_{k-1}."""
    t = np.asarray(t, float)
    t0, t1 = np.ones_like(t), t.copy()
    if k == 0:
        return t0
    for _ in range(1, k):
        t0, t1 = t1, 2 * t * t1 - t0
    return t1


def snap(x, targets):
    """Chỉ số điểm dữ liệu gần nhất với từng mốc lý thuyết (x đã sắp tăng)."""
    idx = np.clip(np.searchsorted(x, targets), 1, len(x) - 1)
    left_closer = np.abs(targets - x[idx - 1]) <= np.abs(x[idx] - targets)
    return np.where(left_closer, idx - 1, idx)


def select_nodes(x, y, n, a, b, kind, xbar=None):
    if n < 1:
        raise ValueError("Bậc n phải ≥ 1.")
    if len(x) < max(n + 1, 2):
        raise ValueError(f"Chỉ có {len(x)} điểm dữ liệu trong [a, b], không đủ cho {n + 1} mốc.")
    info = None
    if kind == "all":
        idx, tg = np.arange(len(x)), x
    elif kind == "nearest":
        if xbar is None:
            raise ValueError("Cần nhập điểm cần tính x̄ để chọn các mốc gần nhất.")
        # trích xuất k = n+1 mốc gần x̄ nhất (với dữ liệu sắp tăng, đó là 1 khối liên tiếp)
        idx = np.sort(np.argsort(np.abs(x - xbar), kind="stable")[:n + 1])
        tg = x[idx]
        info = f"Đã trích xuất k = {n + 1} mốc gần x̄ = {xbar:.6g} nhất: từ {x[idx[0]]:.6g} đến {x[idx[-1]]:.6g}."
    else:
        tg = np.linspace(a, b, n + 1) if kind == "equal" else np.sort(cheb_targets(a, b, n)[1])
        idx = snap(x, tg)
        if len(set(idx.tolist())) < n + 1:
            raise ValueError("Dữ liệu quá thưa: nhiều mốc lý thuyết rơi vào cùng 1 điểm dữ liệu. Hãy giảm n.")
    return dict(x=x[idx], y=y[idx], targets=np.asarray(tg, float), idx=idx,
                dev=float(np.max(np.abs(x[idx] - tg))), info=info)


def lagrange_eval(xn, yn, xq):
    """P_n(x) = w(x) * sum_i (y_i / D_i) / (x - x_i)  (dạng ở slide), tính trên biến chuẩn hoá về [-1,1]."""
    xq = np.atleast_1d(np.asarray(xq, float))
    c, s = (xn.min() + xn.max()) / 2, (xn.max() - xn.min()) / 2 or 1.0
    t, q = (xn - c) / s, (xq - c) / s
    D = np.array([np.prod(t[i] - np.delete(t, i)) for i in range(len(t))])
    diff = q[:, None] - t[None, :]
    with np.errstate(all="ignore"):
        out = np.prod(diff, axis=1) * np.sum((yn / D)[None, :] / diff, axis=1)
    hit = diff == 0
    rows = hit.any(axis=1)
    out[rows] = yn[hit.argmax(axis=1)][rows]
    return out


def dd_max(x, y, k):
    """max |f[x_j..x_{j+k}]| trên các cửa sổ liên tiếp (≈ M_k / k!). None nếu không đủ dữ liệu."""
    if len(x) < k + 1:
        return None
    d = np.asarray(y, float)
    for j in range(1, k + 1):
        d = (d[1:] - d[:-1]) / (x[j:] - x[:-j])
    return float(np.max(np.abs(d)))


def w_abs(xn, xq):
    return np.abs(np.prod(np.atleast_1d(xq)[:, None] - xn[None, :], axis=1))


def fit(xs, ys, a, b, n, kind, xbar=None):
    """Chọn mốc, nội suy, đo sai số thực trên mọi điểm dữ liệu trong [a,b]."""
    nd = select_nodes(xs, ys, n, a, b, kind, xbar)
    xn, yn = nd["x"], nd["y"]
    err = np.abs(ys - lagrange_eval(xn, yn, xs))
    k = int(np.argmax(err))
    grid = np.linspace(a, b, 600)
    return dict(nd=nd, xn=xn, yn=yn, err=err, max=float(err[k]), argmax=float(xs[k]),
                rmse=float(np.sqrt(np.mean(err ** 2))), wsup=float(np.max(w_abs(xn, grid))), grid=grid)


def thin(x, y, m=600):
    s = max(1, len(x) // m)
    return x[::s], y[::s]


def parse_points(text, a, b):
    if not str(text).strip():
        return np.array([(a + b) / 2])
    out = []
    for tok in re.split(r"[;\s]+", str(text).strip()):
        if tok:
            try:
                out.append(float(tok.replace(",", ".")))
            except ValueError:
                raise ValueError(f"Không đọc được '{tok}'. Ngăn cách các điểm bằng dấu ; hoặc xuống dòng.")
    return np.array(out)


def poly_str(c, var="x"):
    return " ".join(f"{'+' if v >= 0 else '-'} {abs(v):.6g}" + (f"·{var}^{i}" if i > 1 else f"·{var}" if i == 1 else "")
                    for i, v in enumerate(c)).lstrip("+ ")
