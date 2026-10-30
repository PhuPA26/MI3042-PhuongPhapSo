"""Nội suy Newton (slide 15): 2 thuật toán — Newton tiến và Newton lùi — mỗi thuật toán hỗ trợ
2 loại mốc: mốc bất kỳ (tỷ sai phân) và mốc cách đều (sai phân hữu hạn)."""
from math import factorial
import numpy as np
from . import register, P
from .core import subset, select_nodes, parse_points, poly_str, dd_max, w_abs
from .interpolation import NODE_OPTS, NODE_NAME, S, ST, vec, fv, data_series, node_note

GRID_OPTS = [["any", "Mốc bất kỳ (tỷ sai phân)"], ["equal", "Mốc cách đều (sai phân hữu hạn)"]]


def dd_columns(xn, yn):
    """cols[k][i] = f[x_i, ..., x_{i+k}]"""
    cols = [np.asarray(yn, float)]
    for k in range(1, len(xn)):
        c = cols[-1]
        cols.append((c[1:] - c[:-1]) / (xn[k:] - xn[:-k]))
    return cols


def newton_eval(xo, a, xq, deriv=False):
    """Sơ đồ Horner cho dạng Newton với các mốc xo theo thứ tự dùng: P = a_n; P = P·(x − x_k) + a_k."""
    xq = np.atleast_1d(np.asarray(xq, float))
    p, dp = np.full_like(xq, a[-1]), np.zeros_like(xq)
    for k in range(len(a) - 2, -1, -1):
        dp = dp * (xq - xo[k]) + p          # quy tắc tích: (p·(x−xₖ))′ = p′·(x−xₖ) + p
        p = p * (xq - xo[k]) + a[k]
    return (p, dp) if deriv else p


def match(xs, ys, xv):
    j = int(np.argmin(np.abs(xs - xv)))
    return j, abs(xs[j] - xv) <= 1e-9 * max(1, abs(xv))


def common(x, y, a, b, p):
    xs, ys = subset(x, y, a, b)
    q = parse_points(p["x_eval"], a, b)
    nd = select_nodes(xs, ys, p["n"], a, b, p["node_type"], q[0])
    return xs, ys, nd, nd["x"], nd["y"], len(nd["x"]) - 1, q


# ------------------------------------------------------------ mốc bất kỳ
def run_any(x, y, a, b, p, forward):
    xs, ys, nd, xn, yn, n, q = common(x, y, a, b, p)
    cols = dd_columns(xn, yn)
    xo = xn if forward else xn[::-1]                              # thứ tự mốc dùng trong công thức
    coef = np.array([c[0] if forward else c[n - k] for k, c in enumerate(cols)])   # f[x₀..xₖ] hoặc f[xₙ..x_{n−k}]
    omega = [np.array([1.0])]
    for k in range(n):
        omega.append(np.polymul(omega[-1], [1.0, -xo[k]]))        # ωₖ = Π_{j<k}(x − x_oⱼ)
    canon = sum(coef[k] * np.concatenate([np.zeros(n - k), omega[k]]) for k in range(n + 1))
    Pq, dPq = newton_eval(xo, coef, q, True)
    om = np.concatenate([[1.0], np.cumprod(q[0] - xo[:-1])])
    omd = np.zeros(n + 1)                                         # ωₖ′(x*): ω_{k+1}′ = ωₖ′·(x−xₖ) + ωₖ
    for k in range(n):
        omd[k + 1] = omd[k] * (q[0] - xo[k]) + om[k]
    terms, termsd = coef * om, coef * omd
    partial, partiald = np.cumsum(terms), np.cumsum(termsd)
    grid = np.linspace(a, b, 600)
    err = np.abs(ys - newton_eval(xo, coef, xs))
    k = int(np.argmax(err))
    dm = dd_max(xs, ys, n + 1)
    wsup = float(np.max(w_abs(xn, grid)))
    lab = "x₀..xₖ" if forward else "xₙ..x_{n−k}"
    vals = []
    for xv, pv, dv in zip(q, Pq, dPq):
        j, hit = match(xs, ys, xv)
        vals.append([xv, pv, dv, ys[j] if hit else None, abs(ys[j] - pv) if hit else None,
                     dm * float(w_abs(xn, xv)[0]) if dm is not None else None])
    notes = [node_note(nd), "P_n(x) = " + poly_str(canon[::-1]) + "   (dạng chính tắc)"]
    if n > 12:
        notes.append("Bậc cao: dạng chính tắc mất nhiều chữ số có nghĩa; hãy dùng dạng Newton để tính giá trị.")
    kind_name = "tiến" if forward else "lùi"
    algo_steps = [
        ST("Chọn / trích xuất các mốc", f"{n + 1} mốc ({NODE_NAME[p['node_type']]}); Newton {kind_name} kết nạp theo thứ tự xₒ.", f"xₒ = {vec(xo)}\nyₒ = {vec(yn if forward else yn[::-1])}"),
        ST("Lập bảng tỷ sai phân", "f[xᵢ, …, x_{i+k}] = (f[xᵢ₊₁, …, x_{i+k}] − f[xᵢ, …, x_{i+k−1}]) / (x_{i+k} − xᵢ).",
           "\n".join(f"cấp {j}: {vec(cols[j])}" for j in range(1, min(n + 1, 5))) + ("\n…" if n > 4 else "")),
        ST(f"Lấy hệ số Newton {kind_name}", "aₖ = f[x₀..xₖ] (đường chéo trên)" if forward else "aₖ = f[xₙ..x_{n−k}] (hàng cuối)", f"aₖ = {vec(coef)}"),
        ST("Lập bảng tích ωₖ(x) = Π_{j<k}(x − xⱼ) và dạng chính tắc", "Nhân dần từng nhân tử (x − xⱼ), rồi P = Σ aₖ·ωₖ.", "P_n(x) = " + poly_str(canon[::-1])),
        ST(f"Cộng dần P_k(x̄) tại x̄ = {fv(q[0])}", "P_k = P_{k−1} + aₖ·ωₖ(x̄).", f"ωₖ(x̄) = {vec(om)}\nP_k(x̄) = {vec(partial)}\nP_n(x̄) = {fv(Pq[0])}"),
        ST("Đạo hàm P_n′(x̄)", "Đạo hàm từng số hạng theo quy tắc tích: ωₖ₊₁′ = ωₖ′·(x − xₖ) + ωₖ.", f"P_k′(x̄) = {vec(partiald)}\nf′(x̄) ≈ {fv(dPq[0])}"),
        ST("Đánh giá sai số", "So với dữ liệu thật trong [a, b].", f"Sai số thực lớn nhất = {fv(err[k])} (tại x = {fv(xs[k])})")]
    return dict(steps=algo_steps, 
        summary=[S("f(x̄) ≈ P_n(x̄)", Pq[0], f"x̄ = {q[0]:.6g}"), S("f′(x̄) ≈ P_n′(x̄)", dPq[0]),
                 S("Bậc n", n), S("Số mốc", n + 1), S("Loại mốc", "Bất kỳ"), S("Kiểu chọn mốc", NODE_NAME[p["node_type"]]),
                 S("Sai số thực lớn nhất", err[k], f"tại x = {xs[k]:.6g}"), S("sup |w_{n+1}(x)|", wsup),
                 S("sup |R_n| ước lượng", None if dm is None else dm * wsup)],
        tables=[
            dict(title="Bảng tỷ sai phân (hàng i: f[x_{i−k}, …, x_i]) — " + ("hệ số tiến là đường chéo trên, " if forward else "hệ số lùi là hàng cuối, ") + "xem bảng dưới",
                 columns=["xᵢ", "yᵢ"] + [f"cấp {k}" for k in range(1, n + 1)],
                 rows=[[xn[i], yn[i]] + [cols[k][i - k] if i >= k else None for k in range(1, n + 1)] for i in range(n + 1)]),
            dict(title=f"Hệ số Newton {'tiến' if forward else 'lùi'}: aₖ = f[{lab}]", columns=["k", "aₖ"],
                 rows=[[k, coef[k]] for k in range(n + 1)]),
            dict(title="Bảng tích: hệ số của ωₖ(x) = Π_{j<k}(x − xⱼ), luỹ thừa giảm dần" + ("" if forward else " (mốc lấy từ xₙ về x₀)"),
                 columns=["k"] + [f"x^{n - j}" for j in range(n + 1)],
                 rows=[[k] + list(np.concatenate([np.zeros(n - k), omega[k]])) for k in range(n + 1)]),
            dict(title=f"Từng lần thêm mốc tại x = {q[0]:.6g}: P_k = P_{{k−1}} + aₖ·ωₖ(x)",
                 columns=["k", "aₖ", "ωₖ(x)", "aₖ·ωₖ(x)", "P_k(x)", "|P_k − P_{k−1}|", "ωₖ′(x)", "aₖ·ωₖ′(x)", "P_k′(x)"],
                 rows=[[k, coef[k], om[k], terms[k], partial[k], None if k == 0 else abs(terms[k]), omd[k], termsd[k], partiald[k]] for k in range(n + 1)]),
            dict(title="Giá trị P_n tại các điểm cần tính",
                 columns=["x", "P_n(x)", "P_n′(x)", "f(x) trong dữ liệu", "|f − P_n|", "Cận |R_n(x)| ước lượng"], rows=vals)],
        plots=[dict(title="f(x) và đa thức Newton P_n(x)", clamp=True,
                    series=[data_series(xs, ys), dict(label="P_n(x)", kind="line", x=grid, y=newton_eval(xo, coef, grid)),
                            dict(label="Mốc nội suy", kind="nodes", x=xn, y=yn)]),
               dict(title=f"P_k(x) tại x = {q[0]:.6g} theo số mốc đã dùng", series=[dict(label="P_k(x)", kind="line", x=list(range(n + 1)), y=partial)])],
        notes=[s for s in notes if s])


# ------------------------------------------------------------ mốc cách đều
def run_equal(x, y, a, b, p, forward):
    xs, ys, nd, xn, yn, n, q = common(x, y, a, b, p)
    d = np.diff(xn)
    if not np.allclose(d, d[0], rtol=1e-6, atol=0):
        raise ValueError("Các mốc không cách đều. Hãy chọn n sao cho các mốc rơi đúng lưới dữ liệu, hoặc đổi sang 'Mốc bất kỳ'.")
    h = float(d.mean())
    D = [np.asarray(yn, float)]                                   # D[k][i] = Δᵏyᵢ
    for k in range(1, n + 1):
        D.append(np.diff(D[-1]))
    dd = [c[0] for c in dd_columns(xn, yn)]

    def ev(xq, fw):
        """Trả về (P(x), P′(x)); t = (x − gốc)/h, P′ = (1/h)·dP/dt."""
        t = (np.atleast_1d(np.asarray(xq, float)) - (xn[0] if fw else xn[-1])) / h
        out = np.full_like(t, D[0][0] if fw else D[0][n])
        dout, pr, dpr = np.zeros_like(t), np.ones_like(t), np.zeros_like(t)
        for k in range(1, n + 1):
            fk = t - (k - 1) if fw else t + (k - 1)
            dpr, pr = dpr * fk + pr, pr * fk
            c = (D[k][0] if fw else D[k][n - k]) / factorial(k)
            out, dout = out + c * pr, dout + c * dpr
        return out, dout / h

    main = lambda xq: ev(xq, forward)[0]
    (Pq, dPq), Pf, Pb = ev(q, forward), ev(q, True)[0], ev(q, False)[0]
    t0 = (q[0] - (xn[0] if forward else xn[-1])) / h
    base = D[0][0] if forward else D[0][n]
    steps, pr, dpr, acc, accd = [[0, base, 1.0, base, base, None, 0.0, 0.0, 0.0]], 1.0, 0.0, base, 0.0
    for k in range(1, n + 1):
        fk = (t0 - (k - 1)) if forward else (t0 + (k - 1))
        dpr, pr = dpr * fk + pr, pr * fk
        dk = D[k][0] if forward else D[k][n - k]
        term, termd = dk / factorial(k) * pr, dk / factorial(k) * dpr / h
        acc, accd = acc + term, accd + termd
        steps.append([k, dk, pr, term, acc, abs(term), dpr, termd, accd])
    grid = np.linspace(a, b, 600)
    err = np.abs(ys - main(xs))
    k = int(np.argmax(err))
    vals = []
    for xv, pv, dv, pf, pb in zip(q, Pq, dPq, Pf, Pb):
        j, hit = match(xs, ys, xv)
        vals.append([xv, pv, dv, pf, pb, ys[j] if hit else None, abs(ys[j] - pv) if hit else None])
    notes = [node_note(nd), f"Bước h = {h:.6g}; t = (x − {'x₀' if forward else 'xₙ'})/h."
             + (" Newton tiến thích hợp khi x gần x₀." if forward else " Newton lùi thích hợp khi x gần xₙ.")]
    if np.any((q < xn[0]) | (q > xn[-1])):
        notes.append("Có điểm nằm ngoài khoảng các mốc: đây là ngoại suy.")
    kname = "tiến" if forward else "lùi"
    algo_steps = [
        ST("Chọn mốc và kiểm tra cách đều", f"{n + 1} mốc ({NODE_NAME[p['node_type']]}), bước h = xᵢ₊₁ − xᵢ.", f"xᵢ = {vec(xn)}\nh = {fv(h)}"),
        ST("Lập bảng sai phân Δᵏy", "Δyᵢ = yᵢ₊₁ − yᵢ, Δᵏyᵢ = Δ(Δᵏ⁻¹yᵢ).", "\n".join(f"Δ{j}y: {vec(D[j])}" for j in range(1, min(n + 1, 5))) + ("\n…" if n > 4 else "")),
        ST(f"Lấy sai phân dùng cho Newton {kname}", "Δᵏy₀ (tiến) hoặc ∇ᵏyₙ = Δᵏy_{n−k} (lùi).", vec([steps[j][1] for j in range(n + 1)])),
        ST("Kiểm tra f[x₀..xₖ] = Δᵏy₀/(k!·hᵏ)", "Liên hệ giữa tỷ sai phân và sai phân.", f"Tỷ sai phân: {vec(dd)}"),
        ST(f"Đặt t = (x̄ − {'x₀' if forward else 'xₙ'})/h", f"x̄ = {fv(q[0])}.", f"t = {fv(t0)}"),
        ST("Cộng từng số hạng", "P_k = P_{k−1} + Δᵏy/k!·t(t∓1)…", f"qₖ(t) = {vec([r_[2] for r_ in steps])}\nP_k(x̄) = {vec([r_[4] for r_ in steps])}"),
        ST("Kết quả và kiểm tra tiến = lùi", "Hai công thức cho cùng 1 đa thức.", f"P_n(x̄) = {fv(Pq[0])}\nP_n′(x̄) = {fv(dPq[0])}\n|P_tiến − P_lùi| = {fv(abs(Pf[0] - Pb[0]))}")]
    return dict(steps=algo_steps, 
        summary=[S("f(x̄) ≈ P_n(x̄)", Pq[0], f"x̄ = {q[0]:.6g}"), S("f′(x̄) ≈ P_n′(x̄)", dPq[0]),
                 S("Bậc n", n), S("Số mốc", n + 1), S("Loại mốc", "Cách đều"), S("Bước h", h), S("t tại điểm đầu", t0),
                 S("Sai số thực lớn nhất", err[k], f"tại x = {xs[k]:.6g}"),
                 S("max |P_tiến − P_lùi|", float(np.max(np.abs(Pf - Pb))), "hai công thức cho cùng 1 đa thức")],
        tables=[
            dict(title="Bảng sai phân hữu hạn Δᵏyᵢ  (∇ᵏyᵢ = Δᵏy_{i−k})", columns=["xᵢ", "yᵢ"] + [f"Δ^{k}" for k in range(1, n + 1)],
                 rows=[[xn[i], yn[i]] + [D[k][i] if i + k <= n else None for k in range(1, n + 1)] for i in range(n + 1)]),
            dict(title="Kiểm tra f[x₀..xₖ] = Δᵏy₀ / (k!·hᵏ)", columns=["k", "Δᵏy₀", "k!·hᵏ", "Δᵏy₀/(k!·hᵏ)", "tỷ sai phân f[x₀..xₖ]"],
                 rows=[[k, D[k][0], factorial(k) * h ** k, D[k][0] / (factorial(k) * h ** k), dd[k]] for k in range(n + 1)]),
            dict(title=f"Từng số hạng tại x = {q[0]:.6g} (Newton {'tiến' if forward else 'lùi'})",
                 columns=["k", "Δᵏy₀" if forward else "∇ᵏyₙ", "qₖ(t)", "số hạng", "P_k(x)", "|số hạng|", "qₖ′(t)", "số hạng đạo hàm", "P_k′(x)"], rows=steps),
            dict(title="Giá trị tại các điểm cần tính",
                 columns=["x", "P_n(x)", "P_n′(x)", "P (tiến)", "P (lùi)", "f(x) trong dữ liệu", "|f − P_n|"], rows=vals)],
        plots=[dict(title="f(x) và đa thức Newton", clamp=True,
                    series=[data_series(xs, ys), dict(label="P_n(x)", kind="line", x=grid, y=main(grid)),
                            dict(label="Mốc nội suy", kind="nodes", x=xn, y=yn)]),
               dict(title=f"P_k(x) tại x = {q[0]:.6g} theo số số hạng đã cộng", series=[dict(label="P_k(x)", kind="line", x=[r[0] for r in steps], y=[r[4] for r in steps])])],
        notes=[s for s in notes if s])


# ------------------------------------------------------------ 2 thuật toán đăng ký
def _params(forward):
    return [P("grid_kind", "Loại mốc", "select", "any", options=GRID_OPTS),
            P("n", "Bậc n (có n+1 mốc)", "int", 5, "Bỏ qua nếu chọn 'mọi điểm dữ liệu'"),
            P("node_type", "Cách lấy mốc từ dữ liệu", "select", "nearest", options=NODE_OPTS),
            P("x_eval", "Điểm cần tính P(x)", "text", "", "Điểm đầu tiên là x̄. Ngăn cách bằng ; — để trống = trung điểm [a,b]")]


def _make(forward):
    def run(x, y, a, b, p):
        return (run_any if p["grid_kind"] == "any" else run_equal)(x, y, a, b, p, forward)
    return run


register("newton_forward", "Nội suy Newton tiến",
         "Dạng tiến từ x₀. Mốc bất kỳ: bảng tỷ sai phân + bảng tích ωₖ + dạng chính tắc. Mốc cách đều: bảng sai phân Δᵏy. "
         "Hiện P_k(x) sau mỗi lần cộng thêm 1 số hạng.", _params(True))(_make(True))
register("newton_backward", "Nội suy Newton lùi",
         "Dạng lùi từ xₙ. Mốc bất kỳ: hệ số f[xₙ..x_{n−k}] ở hàng cuối bảng tỷ sai phân. Mốc cách đều: ∇ᵏyₙ = Δᵏy_{n−k}. "
         "Hiện P_k(x) sau mỗi lần cộng thêm 1 số hạng.", _params(False))(_make(False))
