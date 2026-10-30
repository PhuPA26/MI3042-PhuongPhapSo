"""Nội suy ngược (slide 17): giải f(x) = ȳ khi f cho dạng bộ điểm.
Bước 1: tìm các khoảng cách ly nghiệm [xₖ, xₖ₊₁] (f − ȳ đổi dấu). Bước 2: tìm nghiệm trong từng khoảng bằng
(a) đa thức nội suy của hàm ngược x = f⁻¹(y), hoặc (b) phương pháp lặp từ đa thức Newton tiến / lùi."""
from math import factorial
import numpy as np
from . import register, P
from .core import subset, lagrange_eval
from .newton import dd_columns
from .interpolation import S, ST, vec, fv, data_series

MAX_DETAIL = 6          # số khoảng được in chi tiết từng bước
sub = lambda k: str(k).translate(str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋"))
sup = lambda k: str(k).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def separate(ys, yb):
    """Khoảng cách ly nghiệm: ('node', i) nếu yᵢ = ȳ; ('interval', k) nếu (yₖ − ȳ)(yₖ₊₁ − ȳ) < 0."""
    s, out = ys - yb, []
    for i in range(len(ys)):
        if s[i] == 0:
            out.append(("node", i))
        if i + 1 < len(ys) and s[i] * s[i + 1] < 0:
            out.append(("interval", i))
    return out


# ---------------------------------------------------------- (a) phương pháp hàm ngược
def inverse_one(xs, ys, k, yb, n):
    N = len(xs)
    for m in range(min(n, N - 1), 0, -1):                       # giảm bậc cho tới khi y đơn điệu (có hàm ngược)
        s = int(np.clip(k - (m - 1) // 2, 0, N - 1 - m))
        idx = np.arange(s, s + m + 1)
        d = np.diff(ys[idx])
        if k in idx and k + 1 in idx and (np.all(d > 0) or np.all(d < 0)):
            break
    o = idx[np.argsort(np.abs(ys[idx] - yb), kind="stable")]     # kết nạp mốc gần ȳ nhất trước
    yo, xo = ys[o], xs[o]
    cols = dd_columns(yo, xo)                                    # tỷ sai phân của hàm ngược x = f⁻¹(y)
    coef = np.array([c[0] for c in cols])
    om = np.concatenate([[1.0], np.cumprod(yb - yo[:-1])])
    terms = coef * om
    part = np.cumsum(terms)
    root = float(part[-1])
    resid = abs(float(lagrange_eval(xs[idx], ys[idx], root)[0]) - yb)
    n_dd = len(yo)
    tables = [
        dict(title="Mốc dùng (xếp theo |yᵢ − ȳ| tăng dần) và bảng tỷ sai phân của hàm ngược x = f⁻¹(y)",
             columns=["yᵢ", "xᵢ"] + [f"cấp {j}" for j in range(1, n_dd)],
             rows=[[yo[i], xo[i]] + [cols[j][i - j] if i >= j else None for j in range(1, n_dd)] for i in range(n_dd)]),
        dict(title="Từng lần kết nạp mốc: P_k(ȳ) = P_{k−1}(ȳ) + aₖ·ωₖ(ȳ)",
             columns=["k", "yₖ kết nạp", "xₖ", "aₖ", "ωₖ(ȳ)", "số hạng", "P_k(ȳ)", "|P_k − P_{k−1}|"],
             rows=[[j, yo[j], xo[j], coef[j], om[j], terms[j], part[j], None if j == 0 else abs(terms[j])] for j in range(n_dd)])]
    notes = [f"Đã dùng bậc {n_dd - 1} thay vì {n} (y phải đơn điệu trên các mốc)."] if n_dd - 1 < n else []
    algo_steps = [
        ST("Chọn mốc quanh khoảng", f"Lấy {n_dd} mốc liên tiếp quanh [x{k}, x{k + 1}] sao cho y đơn điệu (điều kiện có hàm ngược); xếp theo |yᵢ − ȳ| tăng dần.", f"yᵢ = {vec(yo)}\nxᵢ = {vec(xo)}"),
        ST("Lập bảng tỷ sai phân của hàm ngược x = f⁻¹(y)", "Nội suy Newton với mốc là yᵢ, giá trị là xᵢ.", "\n".join(f"cấp {j}: {vec(cols[j])}" for j in range(1, min(n_dd, 5)))),
        ST("Kết nạp từng mốc", "P_k(ȳ) = P_{k−1}(ȳ) + aₖ·ωₖ(ȳ).", f"aₖ = {vec(coef)}\nP_k(ȳ) = {vec(part)}"),
        ST("Nghiệm", "x̄ = P_n(ȳ); kiểm tra bằng đa thức nội suy thuận trên cùng các mốc.", f"x̄ = {fv(root)}\n|f(x̄) − ȳ| ≈ {fv(resid)}")]
    return dict(steps=algo_steps, root=root, resid=resid, ok=True, tables=tables, conv=(list(range(n_dd)), part), notes=notes)


# ---------------------------------------------------------- (b) phương pháp lặp
def iterate_one(xs, ys, h, k, yb, n, forward, eps, max_iter):
    N = len(xs)
    m = min(n, N - 1 - k if forward else k + 1)
    D = [np.asarray(ys, float)]
    for _ in range(m):
        D.append(np.diff(D[-1]))
    base = ys[k] if forward else ys[k + 1]
    c = [None] + [(D[j][k] if forward else D[j][k + 1 - j]) / factorial(j) for j in range(1, m + 1)]   # ∇ʲy_{k+1} = Δʲy_{k+1−j}
    sg = -1 if forward else 1

    def poly(t):
        pr, v = 1.0, base
        for j in range(1, m + 1):
            pr *= t + sg * (j - 1)
            v += c[j] * pr
        return v

    def phi(t):
        pr, v = 1.0, 0.0
        for j in range(1, m + 1):
            pr *= t + sg * (j - 1)
            if j >= 2:
                v += c[j] * pr
        return (yb - base) / c[1] - v / c[1]

    x0 = xs[k] if forward else xs[k + 1]
    t = (yb - base) / c[1]
    rows, ok, note = [[0, t, None, x0 + t * h, abs(poly(t) - yb)]], False, None
    for j in range(1, max_iter + 1):
        tn = phi(t)
        if not np.isfinite(tn) or abs(tn) > 1e8:
            note = "Phép lặp phân kỳ (|t| quá lớn): thử giảm bậc n hoặc dùng công thức còn lại (tiến/lùi)."
            break
        rows.append([j, tn, abs(tn - t), x0 + tn * h, abs(poly(tn) - yb)])
        done, t = abs(tn - t) < eps, tn
        if done:
            ok = True
            break
    if not ok and note is None:
        note = f"Chưa đạt sai số {eps:g} sau {max_iter} lần lặp."
    sym = "Δ" if forward else "∇"
    base_lab = f"y{sub(k if forward else k + 1)}"
    tables = [
        dict(title=f"Hệ số dùng cho công thức lặp (Newton {'tiến từ xₖ' if forward else 'lùi từ xₖ₊₁'}, h = {h:.6g})",
             columns=["j", "sai phân", "giá trị", "chia j!"],
             rows=[[0, base_lab, base, base]] + [[j, f"{sym}{sup(j)}y{sub(k if forward else k + 1)}", c[j] * factorial(j), c[j]] for j in range(1, m + 1)]),
        dict(title="Từng lần lặp: t₀ = (ȳ − y)/Δy,  t_{j+1} = φ(t_j),  x̄ = x_gốc + t·h",
             columns=["j", "t_j", "|t_j − t_{j−1}|", "x̄_j = x_gốc + t_j·h", "|P(t_j) − ȳ|"], rows=rows)]
    notes = ([f"Bậc giảm còn {m} vì thiếu mốc về phía {'phải' if forward else 'trái'}."] if m < n else []) + ([note] if note else [])
    rtxt = "\n".join(f"j={r_[0]}: t = {fv(r_[1])}" + ("" if r_[2] is None else f", |Δt| = {fv(r_[2])}") for r_ in rows[:15]) + ("\n…" if len(rows) > 15 else "")
    algo_steps = [
        ST("Tính sai phân tại điểm xuất phát", ("Newton tiến từ xₖ" if forward else "Newton lùi từ xₖ₊₁") + f", bậc {m}, h = {fv(h)}.", f"y_gốc = {fv(base)}\nΔʲy/j! (j = 1..) = {vec(c[1:])}"),
        ST("Giá trị ban đầu t₀", "t₀ = (ȳ − y_gốc)/Δy.", f"t₀ = {fv(rows[0][1])}"),
        ST("Lặp t_{j+1} = φ(t_j)", "φ(t) = t₀ − (1/Δy)·[Δ²y/2!·t(t∓1) + …]; dừng khi |t_{j+1} − t_j| < ε.", rtxt),
        ST("Nghiệm", "x̄ = x_gốc + t·h.", f"x̄ = {fv(rows[-1][3])}  ({'hội tụ' if ok else 'CHƯA hội tụ'})\n|P(t) − ȳ| = {fv(rows[-1][4])}")]
    return dict(steps=algo_steps, root=rows[-1][3], resid=rows[-1][4], ok=ok, tables=tables, conv=([r[0] for r in rows], [r[1] for r in rows]), notes=notes)


# ---------------------------------------------------------- khung chung
def run_inverse(x, y, a, b, p, mode):
    xs, ys = subset(x, y, a, b)
    yb = p.get("y_bar")
    if yb is None:
        raise ValueError("Hãy nhập giá trị ȳ (số thực).")
    n = p["n"]
    if len(xs) < 2 or n < 1:
        raise ValueError("Cần ít nhất 2 điểm trong [a, b] và bậc n ≥ 1.")
    found = separate(ys, yb)
    if not found:
        raise ValueError(f"Không có khoảng cách ly nghiệm: f chỉ nhận giá trị từ {ys.min():.6g} đến {ys.max():.6g} trên [a, b], không đổi dấu f − ȳ.")
    h = None
    if mode != "inverse":
        d = np.diff(xs)
        if not np.allclose(d, d[0], rtol=1e-6, atol=0):
            raise ValueError("Phương pháp lặp cần mốc cách đều trong [a, b].")
        h = float(d.mean())
    algo_steps = [ST("Tìm các khoảng cách ly nghiệm", "Khoảng [xₖ, xₖ₊₁] mà (yₖ − ȳ)(yₖ₊₁ − ȳ) < 0 (hoặc mốc có yᵢ = ȳ).",
                     "\n".join(f"{'mốc x' + str(k) if kd == 'node' else '[x' + str(k) + ', x' + str(k + 1) + ']'} = " +
                               (fv(xs[k]) if kd == 'node' else f"[{fv(xs[k])}, {fv(xs[k + 1])}]") for kd, k in found[:12]))]
    roots, details, notes, conv = [], [], [], []
    for num, (kind, k) in enumerate(found, 1):
        if kind == "node":
            roots.append([num, "trùng mốc", xs[k], None, ys[k], None, xs[k], 0.0])
            continue
        r = (inverse_one(xs, ys, k, yb, n) if mode == "inverse"
             else iterate_one(xs, ys, h, k, yb, n, mode == "forward", p["eps"], p["max_iter"]))
        roots.append([num, "khoảng" if r["ok"] else "khoảng – CHƯA HỘI TỤ", xs[k], xs[k + 1], ys[k], ys[k + 1], r["root"], r["resid"]])
        lab = f"[{xs[k]:.6g}, {xs[k + 1]:.6g}]"
        if len(details) < MAX_DETAIL * 2:
            details += [dict(t, title=f"Khoảng {lab}: " + t["title"]) for t in r["tables"]]
        notes += [f"Khoảng {lab}: {s}" for s in r["notes"]]
        if len(details) <= MAX_DETAIL * 2:
            algo_steps += [dict(st_, title=f"Khoảng {lab} – {st_['title']}") for st_ in r["steps"]]
        conv.append(dict(label=lab, kind="line", x=r["conv"][0], y=r["conv"][1]))
    if len(found) > MAX_DETAIL:
        notes.append(f"Có {len(found)} khoảng; chỉ in chi tiết {MAX_DETAIL} khoảng đầu (kết quả đủ ở bảng đầu).")
    if mode == "inverse":
        notes.append("Điều kiện: f đơn điệu trên các mốc dùng để hàm ngược tồn tại; web tự giảm bậc nếu cần. Sai số kiểm tra |f(x̄) − ȳ| lấy từ đa thức nội suy thuận trên cùng các mốc.")
    else:
        notes.append("Công thức lặp t_{j+1} = φ(t_j) hội tụ khi |φ′(t)| < 1 gần nghiệm. Sai số kiểm tra |P(t) − ȳ| lấy từ đa thức Newton.")
    xr = [r_[6] for r_ in roots]
    summary = [S("ȳ", yb), S("Số khoảng cách ly nghiệm", len(found))] + \
              [S(f"Nghiệm x̄ #{r_[0]}", r_[6], "trùng mốc" if r_[1] == "trùng mốc" else f"trong [{r_[2]:.6g}, {r_[3]:.6g}]" + (" – chưa hội tụ" if "CHƯA" in r_[1] else "")) for r_ in roots[:6]]
    plots = [dict(title="Dữ liệu, đường y = ȳ và các nghiệm", clamp=False,
                  series=[data_series(xs, ys), dict(label="y = ȳ", kind="line", x=[xs[0], xs[-1]], y=[yb, yb]),
                          dict(label="Nghiệm x̄", kind="nodes", x=xr, y=[yb] * len(xr))])]
    if conv:
        plots.append(dict(title="P_k(ȳ) theo số mốc kết nạp" if mode == "inverse" else "t_j theo số lần lặp", series=conv))
    tables = [dict(title="Các khoảng cách ly nghiệm và nghiệm tìm được",
                   columns=["#", "loại", "xₖ", "xₖ₊₁", "yₖ", "yₖ₊₁", "nghiệm x̄", "|f(x̄) − ȳ| kiểm tra"], rows=roots)] + details
    return dict(steps=algo_steps, summary=summary, tables=tables, plots=plots, notes=notes)


YB = P("y_bar", "Giá trị ȳ (giải f(x) = ȳ)", "number", 3.0)
N_ = P("n", "Bậc đa thức n", "int", 3, "Số mốc dùng = n + 1 quanh mỗi khoảng")
EPS = P("eps", "Sai số dừng ε", "number", 1e-8, "Dừng khi |t_{j+1} − t_j| < ε")
IT = P("max_iter", "Số lần lặp tối đa", "int", 100)

register("inverse_function", "Nội suy ngược – phương pháp hàm ngược",
         "Tìm khoảng cách ly nghiệm của f(x) = ȳ rồi nội suy hàm ngược x = f⁻¹(y) qua các mốc (yᵢ, xᵢ); hiện từng lần kết nạp mốc.",
         [YB, N_])(lambda x, y, a, b, p: run_inverse(x, y, a, b, p, "inverse"))
register("inverse_iter_forward", "Nội suy ngược – phương pháp lặp (Newton tiến)",
         "Mốc cách đều. Công thức lặp t_{j+1} = φ(t_j) từ đa thức Newton tiến xuất phát từ xₖ; hiện kết quả từng lần lặp.",
         [YB, N_, EPS, IT])(lambda x, y, a, b, p: run_inverse(x, y, a, b, p, "forward"))
register("inverse_iter_backward", "Nội suy ngược – phương pháp lặp (Newton lùi)",
         "Mốc cách đều. Công thức lặp t_{j+1} = ψ(t_j) từ đa thức Newton lùi xuất phát từ xₖ₊₁; hiện kết quả từng lần lặp.",
         [YB, N_, EPS, IT])(lambda x, y, a, b, p: run_inverse(x, y, a, b, p, "backward"))
