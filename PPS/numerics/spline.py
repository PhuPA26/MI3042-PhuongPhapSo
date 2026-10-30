"""Hàm ghép trơn – Spline (slide 18): tuyến tính, cấp 2, cấp 3. Chỉ số 0..N−1 (slide đánh số 1..n)."""
import numpy as np
from numpy.polynomial import Polynomial as Poly
from . import register, P
from .core import subset, parse_points, thin
from .interpolation import S, ST, vec, fv, data_series

MAXR = 20      # số dòng tối đa in ra ở các bảng dài


def thomas(lo, di, up, rhs):
    """Giải hệ tam đường chéo; trả về nghiệm và các hệ số quét thuận c', d'."""
    n = len(di)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = (up[0] / di[0] if n > 1 else 0.0), rhs[0] / di[0]
    for i in range(1, n):
        den = di[i] - lo[i] * cp[i - 1]
        cp[i] = up[i] / den if i < n - 1 else 0.0
        dp[i] = (rhs[i] - lo[i] * dp[i - 1]) / den
    sol = np.zeros(n)
    sol[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        sol[i] = dp[i] - cp[i] * sol[i + 1]
    return sol, cp, dp


def piece(kind, xs, ys, par, k, xq):
    """(S, S′, S″) của đoạn thứ k tại xq."""
    x0, x1, h = xs[k], xs[k + 1], xs[k + 1] - xs[k]
    xq = np.asarray(xq, float)
    if kind == 1:
        s = (ys[k + 1] - ys[k]) / h
        return ys[k] + s * (xq - x0), s + 0 * xq, 0 * xq
    if kind == 2:
        m0, m1 = par[k], par[k + 1]
        return ((-m0 * (xq - x1) ** 2 + m1 * (xq - x0) ** 2) / (2 * h) + ys[k] + m0 * h / 2,
                (-m0 * (xq - x1) + m1 * (xq - x0)) / h, (m1 - m0) / h + 0 * xq)
    a0, a1 = par[k], par[k + 1]
    th, be = ys[k] / h - a0 * h / 6, ys[k + 1] / h - a1 * h / 6
    return ((-a0 * (xq - x1) ** 3 + a1 * (xq - x0) ** 3) / (6 * h) + be * (xq - x0) - th * (xq - x1),
            (-a0 * (xq - x1) ** 2 + a1 * (xq - x0) ** 2) / (2 * h) + be - th,
            (-a0 * (xq - x1) + a1 * (xq - x0)) / h)


def evaluate(kind, xs, ys, par, xq):
    xq = np.atleast_1d(np.asarray(xq, float))
    ks = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, len(xs) - 2)
    out = np.zeros((3, len(xq)))
    for k in np.unique(ks):
        m = ks == k
        for r, v in enumerate(piece(kind, xs, ys, par, k, xq[m])):
            out[r, m] = v
    return out


def power_coefs(kind, xs, ys, par, k):
    X, x0, x1, h = Poly([0, 1]), xs[k], xs[k + 1], xs[k + 1] - xs[k]
    if kind == 1:
        p = ys[k] + (ys[k + 1] - ys[k]) / h * (X - x0)
    elif kind == 2:
        p = (-par[k] * (X - x1) ** 2 + par[k + 1] * (X - x0) ** 2) / (2 * h) + ys[k] + par[k] * h / 2
    else:
        th, be = ys[k] / h - par[k] * h / 6, ys[k + 1] / h - par[k + 1] * h / 6
        p = (-par[k] * (X - x1) ** 3 + par[k + 1] * (X - x0) ** 3) / (6 * h) + be * (X - x0) - th * (X - x1)
    c = np.zeros(kind + 1)
    c[:len(p.coef)] = p.coef
    return c[::-1]                                              # giảm dần: aₖ, bₖ, …


def run_spline(x, y, a, b, p, kind):
    xs, ys = subset(x, y, a, b)
    N = len(xs)
    if N < 3:
        raise ValueError("Cần ít nhất 3 điểm trong [a, b].")
    h = np.diff(xs)
    tables, notes, par, bc = [], [], None, "—"
    if kind == 2:
        g = 2 * np.diff(ys) / h
        m = np.zeros(N)
        m[0] = p["m0"] if p["m0"] is not None else 0.0
        rows = []
        for k in range(N - 1):
            m[k + 1] = g[k] - m[k]
            rows.append([k, xs[k], h[k], g[k], m[k], m[k + 1]])
        par, bc = m, f"m₀ = S′(x₀) = {m[0]:.6g}"
        tables.append(dict(title="Hệ n−1 phương trình, n ẩn: m_{k+1} + m_k = γₖ = 2(y_{k+1} − yₖ)/hₖ — thế lần lượt từ m₀",
                           columns=["k", "xₖ", "hₖ", "γₖ", "mₖ", "m_{k+1}"], rows=rows[:MAXR]))
        notes.append("Hệ có n−1 phương trình cho n ẩn nên phải cho thêm 1 điều kiện (ở đây là m₀ = S′(x₀)).")
    elif kind == 3:
        gam = np.diff(ys[1:]) / h[1:] - np.diff(ys[:-1]) / h[:-1]          # γ_j, j = 1..N−2
        clamped = p["bc"] == "clamped"
        lo, di, up, rhs = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        for j in range(1, N - 1):
            lo[j], di[j], up[j], rhs[j] = h[j - 1] / 6, (h[j - 1] + h[j]) / 3, h[j] / 6, gam[j - 1]
        if clamped:
            di[0], up[0], rhs[0] = h[0] / 3, h[0] / 6, (ys[1] - ys[0]) / h[0] - p["d0"]
            lo[-1], di[-1], rhs[-1] = h[-1] / 6, h[-1] / 3, p["dn"] - (ys[-1] - ys[-2]) / h[-1]
            sol, cp, dp = thomas(lo, di, up, rhs)
            r0, bc = 0, f"kẹp: S′(x₀) = {p['d0']:.6g}, S′(xₙ) = {p['dn']:.6g}"
        else:
            sol_in, cp_in, dp_in = thomas(lo[1:-1], di[1:-1], up[1:-1], rhs[1:-1])
            sol, cp, dp = np.concatenate([[0], sol_in, [0]]), np.concatenate([[0], cp_in, [0]]), np.concatenate([[0], dp_in, [0]])
            r0, bc = 1, "tự nhiên: α₀ = αₙ = 0"
        par = sol
        lab = lambda i: f"α{i}".translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
        tables.append(dict(title="Hệ phương trình tam đường chéo: (h_{j−1}/6)α_{j−1} + ((h_{j−1}+h_j)/3)αⱼ + (hⱼ/6)α_{j+1} = γⱼ  (quét thuận Thomas)",
                           columns=["j", "dưới", "chéo", "trên", "vế phải γⱼ", "c′ⱼ", "d′ⱼ", "nghiệm αⱼ"],
                           rows=[[j, lo[j] if j > r0 else None, di[j], up[j] if j < N - 1 - r0 else None, rhs[j], cp[j], dp[j], sol[j]]
                                 for j in range(r0, N - r0)][:MAXR]))
        if N - 2 * r0 <= 12:
            A = np.zeros((N - 2 * r0, N - 2 * r0))
            for i, j in enumerate(range(r0, N - r0)):
                A[i, i] = di[j]
                if i > 0:
                    A[i, i - 1] = lo[j]
                if i < A.shape[0] - 1:
                    A[i, i + 1] = up[j]
            tables.append(dict(title="Ma trận hệ (dạng đầy đủ)", columns=[lab(j) for j in range(r0, N - r0)] + ["γ"],
                               rows=[list(A[i]) + [rhs[r0 + i]] for i in range(A.shape[0])]))
        notes.append("Hệ có n−2 phương trình cho n ẩn nên phải cho thêm 2 điều kiện biên: " + bc + ".")
    names = ["aₖ·x + bₖ", "aₖ·x² + bₖ·x + cₖ", "aₖ·x³ + bₖ·x² + cₖ·x + dₖ"][kind - 1]
    cn = ["aₖ", "bₖ", "cₖ", "dₖ"][:kind + 1]
    tables.append(dict(title=f"Hệ số từng đoạn: Sₖ(x) = {names}" + ("  (khai triển từ dạng ổn định; dùng dạng ổn định để tính)" if kind > 1 else ""),
                       columns=["k", "xₖ", "xₖ₊₁", "hₖ"] + cn,
                       rows=[[k, xs[k], xs[k + 1], h[k]] + list(power_coefs(kind, xs, ys, par, k)) for k in range(min(N - 1, MAXR))]))
    # kiểm tra tính trơn
    at_nodes = evaluate(kind, xs, ys, par, xs)[0]
    j1 = j2 = 0.0
    for k in range(N - 2):
        L, R = piece(kind, xs, ys, par, k, xs[k + 1]), piece(kind, xs, ys, par, k + 1, xs[k + 1])
        j1, j2 = max(j1, abs(L[1] - R[1])), max(j2, abs(L[2] - R[2]))
    q = parse_points(p["x_eval"], xs[0], xs[-1])
    Sq = evaluate(kind, xs, ys, par, q)
    vals = []
    for i, xv in enumerate(q):
        j = int(np.argmin(np.abs(xs - xv)))
        hit = abs(xs[j] - xv) <= 1e-9 * max(1, abs(xv))
        vals.append([xv, Sq[0, i], Sq[1, i], Sq[2, i], ys[j] if hit else None, abs(ys[j] - Sq[0, i]) if hit else None])
    tables.append(dict(title="Giá trị tại các điểm cần tính", columns=["x", "S(x)", "S′(x)", "S″(x)", "f(x) trong dữ liệu", "|f − S|"], rows=vals))
    grid = np.linspace(xs[0], xs[-1], 800)
    G = evaluate(kind, xs, ys, par, grid)
    plots = [dict(title=f"Spline {['tuyến tính', 'cấp 2', 'cấp 3'][kind - 1]} và dữ liệu", clamp=False,
                  series=[data_series(xs, ys), dict(label="S(x)", kind="line", x=grid, y=G[0]), dict(label="Mốc", kind="nodes", x=xs, y=ys)])]
    if kind >= 2:
        plots.append(dict(title=f"Đạo hàm cấp {kind - 1} của S (liên tục)", series=[dict(label=f"S^({kind - 1})(x)", kind="line", x=grid, y=G[kind - 1])]))
    if np.any((q < xs[0]) | (q > xs[-1])):
        notes.append("Có điểm nằm ngoài [x₀, xₙ]: spline chỉ xác định trong khoảng các mốc (đang ngoại suy bằng đoạn đầu/cuối).")
    summary = [S("Số mốc", N), S("Số đoạn", N - 1), S("Điều kiện thêm", bc), S("S(x̄)", Sq[0, 0], f"x̄ = {q[0]:.6g}"),
               S("S′(x̄)", Sq[1, 0]), S("max |S(xᵢ) − yᵢ|", float(np.max(np.abs(at_nodes - ys))), "phải ≈ 0"),
               S("Độ nhảy lớn nhất của S′ tại mốc trong", j1, "phải ≈ 0 (cấp ≥ 2)" if kind >= 2 else "spline tuyến tính không cần trơn")]
    if kind == 3:
        summary.append(S("Độ nhảy lớn nhất của S″", j2, "phải ≈ 0"))
    kn = ["tuyến tính", "cấp 2", "cấp 3"][kind - 1]
    algo_steps = [ST("Tính hₖ = x_{k+1} − xₖ", f"{N} mốc ⇒ {N - 1} đoạn.", f"hₖ = {vec(h)}")]
    if kind == 1:
        algo_steps.append(ST("Hệ số từng đoạn Sₖ = aₖx + bₖ", "aₖ = (y_{k+1} − yₖ)/hₖ,  bₖ = (yₖ·x_{k+1} − y_{k+1}·xₖ)/hₖ.",
                             f"aₖ = {vec(np.diff(ys) / h)}\nbₖ = {vec((ys[:-1] * xs[1:] - ys[1:] * xs[:-1]) / h)}"))
    elif kind == 2:
        algo_steps += [ST("Tính γₖ = 2(y_{k+1} − yₖ)/hₖ", "Vế phải của hệ m_{k+1} + mₖ = γₖ (n−1 phương trình, n ẩn).", f"γₖ = {vec(g)}"),
                       ST("Thế lần lượt từ m₀", f"Cho thêm điều kiện m₀ = {fv(par[0])}; m_{{k+1}} = γₖ − mₖ.", f"mₖ = {vec(par)}"),
                       ST("Hệ số từng đoạn Sₖ(x) = aₖx² + bₖx + cₖ", "Từ Sₖ = [−mₖ(x−x_{k+1})² + m_{k+1}(x−xₖ)²]/(2hₖ) + yₖ + mₖhₖ/2.", "xem bảng 'Hệ số từng đoạn'")]
    else:
        algo_steps += [ST("Tính γⱼ = (y_{j+1}−yⱼ)/hⱼ − (yⱼ−y_{j−1})/h_{j−1}", "Vế phải của hệ tam đường chéo (n−2 phương trình, n ẩn).", f"γⱼ = {vec(gam)}"),
                       ST("Lập hệ (h_{j−1}/6)α_{j−1} + ((h_{j−1}+h_j)/3)αⱼ + (hⱼ/6)α_{j+1} = γⱼ", "Thêm điều kiện biên: " + bc + ".", "xem bảng hệ phương trình"),
                       ST("Giải hệ bằng thuật toán Thomas", "Quét thuận tính c′ⱼ, d′ⱼ; quét ngược tìm αⱼ = S″(xⱼ).", f"αⱼ = {vec(par)}"),
                       ST("Hệ số từng đoạn Sₖ(x) = aₖx³ + bₖx² + cₖx + dₖ", "Từ dạng Sₖ = [−αₖ(x−x_{k+1})³ + α_{k+1}(x−xₖ)³]/(6hₖ) + β̄ₖ(x−xₖ) − θ̄ₖ(x−x_{k+1}).", "xem bảng 'Hệ số từng đoạn'")]
    algo_steps += [ST("Kiểm tra tính trơn", "S đi qua các mốc; S′ (và S″ với cấp 3) liên tục tại mốc trong.",
                      f"max|S(xᵢ) − yᵢ| = {fv(float(np.max(np.abs(at_nodes - ys))))}\nđộ nhảy lớn nhất của S′ = {fv(j1)}" + (f"\nđộ nhảy lớn nhất của S″ = {fv(j2)}" if kind == 3 else "")),
                   ST(f"Tính S(x̄) tại x̄ = {fv(q[0])}", "Xác định đoạn chứa x̄ rồi tính bằng dạng ổn định của Sₖ.", f"S(x̄) = {fv(Sq[0, 0])}\nS′(x̄) = {fv(Sq[1, 0])}\nS″(x̄) = {fv(Sq[2, 0])}")]
    return dict(steps=algo_steps, summary=summary, tables=tables, plots=plots, notes=notes)


XE = P("x_eval", "Điểm cần tính S(x)", "text", "", "Ngăn cách bằng ; — để trống = trung điểm")
register("spline_1", "Spline tuyến tính (hàm ghép bậc 1)", "Nối các mốc bằng đoạn thẳng Sₖ(x) = aₖx + bₖ; hiện hệ số từng đoạn.", [XE])(
    lambda x, y, a, b, p: run_spline(x, y, a, b, p, 1))
register("spline_2", "Spline cấp 2 (hàm ghép bậc 2)",
         "Đa thức bậc 2 từng đoạn, S trơn cấp 1; hệ n−1 phương trình giải bằng thế lần lượt từ m₀ — hiện từng bước.",
         [P("m0", "Điều kiện thêm m₀ = S′(x₀)", "number", 0.0), XE])(lambda x, y, a, b, p: run_spline(x, y, a, b, p, 2))
register("spline_3", "Spline cấp 3 (hàm ghép bậc 3)",
         "Đa thức bậc 3 từng đoạn, S trơn cấp 2; giải hệ tam đường chéo theo αₖ = S″(xₖ) — hiện bảng quét Thomas.",
         [P("bc", "Điều kiện biên", "select", "natural", options=[["natural", "Tự nhiên: S″(x₀) = S″(xₙ) = 0"], ["clamped", "Kẹp: cho S′(x₀), S′(xₙ)"]]),
          P("d0", "S′(x₀) (khi chọn 'kẹp')", "number", 0.0), P("dn", "S′(xₙ) (khi chọn 'kẹp')", "number", 0.0), XE])(
    lambda x, y, a, b, p: run_spline(x, y, a, b, p, 3))
