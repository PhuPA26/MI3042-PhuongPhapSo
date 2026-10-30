"""Nội suy trung tâm (slide 16): Gauss I, Gauss II, Stirling, Bessel — mốc cách đều.
Mốc được trích xuất quanh x̄ rồi kết nạp xen kẽ về hai phía từ mốc trung tâm x₀."""
from math import factorial
import numpy as np
from . import register, P
from .core import subset, parse_points, dd_max, w_abs
from .interpolation import S, ST, vec, fv, data_series

SUB = str.maketrans("0123456789-", "₀₁₂₃₄₅₆₇₈₉₋")
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
sub = lambda k: str(k).translate(SUB)
sup = lambda k: str(k).translate(SUP)


def engine(y, c, J, kind, t):
    """Chuỗi Gauss với mốc trung tâm ở chỉ số c. kind 1: kết nạp x₀,x₁,x₋₁,x₂,… ; kind 2: x₀,x₋₁,x₁,x₋₂,…
    Số hạng j: Δʲy_{c−⌊j/2⌋} (I) hoặc Δʲy_{c−⌈j/2⌉} (II) chia j!, nhân qⱼ(t) = Π_{i<j}(t − sᵢ)."""
    t = np.atleast_1d(np.asarray(t, float))
    diff = [np.asarray(y, float)]
    for j in range(1, J + 1):
        diff.append(np.diff(diff[-1]))

    def s(i):
        if i == 0:
            return 0
        v = (i + 1) // 2 if i % 2 else -(i // 2)
        return v if kind == 1 else -v

    q, qd, steps = np.ones_like(t), np.zeros_like(t), []
    for j in range(J + 1):
        start = c
        if j > 0:
            sj = s(j - 1)
            qd, q = qd * (t - sj) + q, q * (t - sj)
            start = c - (j // 2 if kind == 1 else (j + 1) // 2)
        if start < 0 or start + j > len(y) - 1:
            raise ValueError("Không đủ mốc quanh x̄ cho bậc đã chọn.")
        D = diff[j][start]
        steps.append(dict(j=j, s=s(j), start=start - c, D=D, w=D / factorial(j), q=q, qd=qd, term=D / factorial(j) * q,
                          dterm=D / factorial(j) * qd))
    return steps


def build(fid, yn, cc, n, t):
    t = np.atleast_1d(np.asarray(t, float))
    if fid == "gauss1":
        return [(engine(yn, cc, n, 1, t), 0)]
    if fid == "gauss2":
        return [(engine(yn, cc, n, 2, t), 0)]
    if fid == "stirling":
        return [(engine(yn, cc, n, 1, t), 0), (engine(yn, cc, n, 2, t), 0)]
    return [(engine(yn, cc, n, 1, t), 0), (engine(yn, cc + 1, n, 2, t - 1), 1)]


def totals(ser, h):
    Pv = sum(sum(st["term"] for st in sts) for sts, _ in ser) / len(ser)
    dP = sum(sum(st["dterm"] for st in sts) for sts, _ in ser) / len(ser) / h
    return Pv, dP


def step_rows(sts, off, h):
    rows, Pj, dPj = [], 0.0, 0.0
    for st in sts:
        Pj, dPj = Pj + st["term"][0], dPj + st["dterm"][0] / h
        j, k0 = st["j"], st["start"] + off
        lab = f"y{sub(k0)}" if j == 0 else f"Δ{sup(j)}y{sub(k0)}"
        rows.append([j, f"x{sub(st['s'] + off)}", lab, st["D"], st["w"], st["q"][0], st["term"][0], Pj,
                     abs(st["term"][0]), st["qd"][0], st["dterm"][0] / h, dPj])
    return rows


STEP_COLS = ["j", "mốc kết nạp", "sai phân dùng", "giá trị", "chia j!", "qⱼ(t)", "số hạng", "P_j(x̄)", "|số hạng|",
             "qⱼ′(t)", "số hạng đạo hàm", "P_j′(x̄)"]


def run_central(x, y, a, b, p, fid):
    xs, ys = subset(x, y, a, b)
    N, n = len(xs), p["n"]
    d = np.diff(xs)
    if N < 3 or not np.allclose(d, d[0], rtol=1e-6, atol=0):
        raise ValueError("Nội suy trung tâm cần dữ liệu cách đều trong [a, b]. Hãy thu hẹp [a, b] hoặc kiểm tra file.")
    h = float(d.mean())
    q = parse_points(p["x_eval"], a, b)
    xb = float(q[0])
    even_nodes = fid == "bessel"
    if even_nodes and n % 2 == 0:
        raise ValueError("Bessel dùng số mốc chẵn: hãy chọn n lẻ (ví dụ n = 3, 5, 7).")
    if not even_nodes and n % 2 == 1:
        raise ValueError("Công thức này dùng số mốc lẻ: hãy chọn n chẵn (ví dụ n = 2, 4, 8).")
    if n < 2 and not even_nodes or n < 1 or N < n + 1:
        raise ValueError(f"Không đủ dữ liệu cho {n + 1} mốc trong [a, b].")
    m = (n - 1) // 2 if even_nodes else n // 2
    ideal = int(np.floor((xb - xs[0]) / h + 1e-9)) if even_nodes else int(round((xb - xs[0]) / h))
    c = int(np.clip(ideal, m, N - 1 - m - (1 if even_nodes else 0)))
    lo, hi = c - m, c + m + (1 if even_nodes else 0)
    xn, yn, cc = xs[lo:hi + 1], ys[lo:hi + 1], c - lo
    xc, t = xn[cc], (q - xn[cc]) / h
    ser = build(fid, yn, cc, n, t)
    Pq, dPq = totals(ser, h)
    # sai số thực trong khoảng các mốc
    mask = (xs >= xn[0] - 1e-9) & (xs <= xn[-1] + 1e-9)
    Pm, _ = totals(build(fid, yn, cc, n, (xs[mask] - xc) / h), h)
    err = np.abs(ys[mask] - Pm)
    k = int(np.argmax(err))
    grid = np.linspace(xn[0], xn[-1], 400)
    Pg, _ = totals(build(fid, yn, cc, n, (grid - xc) / h), h)
    dm = dd_max(xs, ys, n + 1)
    vals = []
    for xv, pv, dv in zip(q, Pq, dPq):
        j = int(np.argmin(np.abs(xs - xv)))
        hit = abs(xs[j] - xv) <= 1e-9 * max(1, abs(xv))
        vals.append([xv, pv, dv, ys[j] if hit else None, abs(ys[j] - pv) if hit else None,
                     dm * float(w_abs(xn, xv)[0]) if dm is not None else None])
    dtab = [yn.astype(float)]
    for j in range(1, n + 1):
        dtab.append(np.diff(dtab[-1]))
    tables = [
        dict(title=f"Các mốc trích xuất (k = {n + 1} mốc quanh x̄; x₀ = {xc:.6g}, h = {h:.6g})", columns=["k", "xₖ", "yₖ"],
             rows=[[i - cc, xn[i], yn[i]] for i in range(len(xn))]),
        dict(title="Bảng sai phân Δʲyₖ", columns=["k", "xₖ", "yₖ"] + [f"Δ{sup(j)}" for j in range(1, n + 1)],
             rows=[[i - cc, xn[i], yn[i]] + [dtab[j][i] if i + j <= n else None for j in range(1, n + 1)] for i in range(n + 1)])]
    names = {"gauss1": ["Gauss I"], "gauss2": ["Gauss II"], "stirling": ["Gauss I", "Gauss II"], "bessel": ["Gauss I", "Gauss II"]}[fid]
    for (sts, off), nm in zip(ser, names):
        tables.append(dict(title=f"Từng lần kết nạp mốc – {nm} tại x̄ = {xb:.6g} (t = {t[0]:.6g})", columns=STEP_COLS,
                           rows=step_rows(sts, off, h)))
    summary = [S("f(x̄) ≈ P(x̄)", Pq[0], f"x̄ = {xb:.6g}"), S("f′(x̄) ≈ P′(x̄)", dPq[0]), S("Bậc n", n), S("Số mốc", n + 1),
               S("Mốc trung tâm x₀", xc), S("t = (x̄ − x₀)/h", t[0]), S("Sai số thực lớn nhất", err[k], f"tại x = {xs[mask][k]:.6g}")]
    plots = [dict(title="f(x) và đa thức nội suy trung tâm", clamp=False,
                  series=[data_series(xs[mask], ys[mask]), dict(label="P(x)", kind="line", x=grid, y=Pg),
                          dict(label="Mốc nội suy", kind="nodes", x=xn, y=yn)])]
    if len(ser) == 2:                                           # Stirling / Bessel: trung bình hai chuỗi
        T = [sts for sts, _ in ser]
        avg, avgd, rows = 0.0, 0.0, []
        for j in range(n + 1):
            ta, tb = T[0][j]["term"][0], T[1][j]["term"][0]
            da, db = T[0][j]["dterm"][0], T[1][j]["dterm"][0]
            avg, avgd = avg + (ta + tb) / 2, avgd + (da + db) / 2 / h
            rows.append([j, ta, tb, (ta + tb) / 2, avg, avgd])
        tables.append(dict(title=f"Trung bình hai chuỗi → {'Stirling' if fid == 'stirling' else 'Bessel'}",
                           columns=["j", "số hạng Gauss I", "số hạng Gauss II", "số hạng trung bình", "P_j(x̄)", "P_j′(x̄)"], rows=rows))
        P1, _ = totals([build(fid, yn, cc, n, (xs[mask] - xc) / h)[0]], h)
        P2, _ = totals([build(fid, yn, cc, n, (xs[mask] - xc) / h)[1]], h)
        summary.append(S("max |P_GaussI − P_GaussII|", float(np.max(np.abs(P1 - P2))), "cùng nội suy các mốc ⇒ cùng 1 đa thức"))
        plots.append(dict(title=f"P_j(x̄) theo số mốc đã kết nạp (trung bình)", series=[dict(label="P_j(x̄)", kind="line", x=[r[0] for r in rows], y=[r[4] for r in rows])]))
    else:
        rows = step_rows(ser[0][0], 0, h)
        plots.append(dict(title="P_j(x̄) theo số mốc đã kết nạp", series=[dict(label="P_j(x̄)", kind="line", x=[r[0] for r in rows], y=[r[7] for r in rows])]))
    tables.append(dict(title="Giá trị tại các điểm cần tính", columns=["x", "P(x)", "P′(x)", "f(x) trong dữ liệu", "|f − P|", "Cận |R_n(x)| ước lượng"], rows=vals))
    hint = {"gauss1": "Gauss I thích hợp khi x̄ ≥ x₀ (0 ≤ t < 1).", "gauss2": "Gauss II thích hợp khi x̄ ≤ x₀ (−1 < t ≤ 0).",
            "stirling": "Stirling hiệu quả khi |t| nhỏ (khoảng |t| ≤ 1/4).", "bessel": "Bessel hiệu quả khi t gần 1/2 (khoảng 1/4 ≤ t ≤ 3/4)."}[fid]
    notes = [f"Mốc được trích xuất quanh x̄ rồi kết nạp xen kẽ hai phía từ x₀. {hint}"]
    if c != ideal:
        notes.append("x̄ gần biên dữ liệu: các mốc không còn đối xứng quanh x̄, |t| lớn hơn nên sai số tăng.")
    if np.any((q < xn[0]) | (q > xn[-1])):
        notes.append("Có điểm nằm ngoài khoảng các mốc: đây là ngoại suy.")
    algo_steps = [
        ST("Trích xuất các mốc quanh x̄", f"Lấy {n + 1} mốc gần x̄ = {fv(xb)} nhất, chọn mốc trung tâm x₀ và bước h.", f"xₖ = {vec(xn)}\nx₀ = {fv(xc)}, h = {fv(h)}"),
        ST("Lập bảng sai phân Δʲyₖ", "Δyₖ = yₖ₊₁ − yₖ, rồi lấy sai phân liên tiếp.", "\n".join(f"Δ{j}y: {vec(dtab[j])}" for j in range(1, min(n + 1, 5))) + ("\n…" if n > 4 else "")),
        ST("Tính t = (x̄ − x₀)/h", "Bessel dùng t cho Gauss I và t − 1 cho Gauss II." if fid == "bessel" else "Biến chuẩn hoá quanh mốc trung tâm.", f"t = {fv(t[0])}")]
    for (sts, off), nm in zip(ser, names):
        algo_steps.append(ST(f"Kết nạp mốc xen kẽ – {nm}", "P_j = P_{j−1} + Δʲy/j!·Π(t − sᵢ); mốc kết nạp: " + "[" + ", ".join(f"x{sub(st['s'] + off)}" for st in sts[:8]) + (", …" if len(sts) > 8 else "") + "]",
                             f"Sai phân dùng = {vec([st['D'] for st in sts])}\nP_j(x̄) = {vec(np.cumsum([st['term'][0] for st in sts]))}"))
    if len(ser) == 2:
        algo_steps.append(ST("Lấy trung bình hai chuỗi", "Stirling / Bessel = (Gauss I + Gauss II)/2, cộng từng cặp số hạng.", f"P_j(x̄) = {vec([r_[4] for r_ in rows])}"))
    algo_steps.append(ST("Kết quả", "P(x̄) và đạo hàm P′(x̄).", f"f(x̄) ≈ {fv(Pq[0])}\nf′(x̄) ≈ {fv(dPq[0])}"))
    return dict(steps=algo_steps, summary=summary, tables=tables, plots=plots, notes=notes)


FORMULAS = [("stirling", "Nội suy Stirling", "Trung bình của Gauss I và Gauss II (số mốc lẻ): triển khai đồng thời về hai phía."),
            ("bessel", "Nội suy Bessel", "Trung bình Gauss I (gốc x₀) và Gauss II (gốc x₁), số mốc chẵn.")]


def _make(fid):
    return lambda x, y, a, b, p: run_central(x, y, a, b, p, fid)


for fid, name, desc in FORMULAS:
    register(fid, name, desc + " Mốc được trích xuất gần x̄ nhất; hiện kết quả từng lần kết nạp mốc và f′(x̄).",
             [P("n", "Bậc n (có n+1 mốc)", "int", 3 if fid == "bessel" else 4,
                "n lẻ (số mốc chẵn)" if fid == "bessel" else "n chẵn (số mốc lẻ)"),
              P("x_eval", "Điểm cần tính P(x)", "text", "", "Điểm đầu tiên là x̄. Ngăn cách bằng ;")])(_make(fid))
