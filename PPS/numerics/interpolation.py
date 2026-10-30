"""Các thuật toán trong 2 slide: Xấp xỉ hàm bằng đa thức nội suy + Đa thức nội suy Lagrange."""
import re
import numpy as np
from . import register, P
from .core import (subset, cheb_targets, cheb_T, lagrange_eval, dd_max, w_abs, fit, thin,
                   parse_points, poly_str, select_nodes)

NODE_OPTS = [["equal", "Mốc cách đều"], ["chebyshev", "Mốc Chebyshev (tối ưu)"],
             ["all", "Mọi điểm dữ liệu trong [a, b]"],
            ["nearest", "Trích xuất k = n+1 điểm gần x̄ nhất"]]
NODE_NAME = dict(NODE_OPTS)


def fv(v, d=8):
    if v is None:
        return "—"
    return str(v) if isinstance(v, (int, np.integer)) else f"{float(v):.{d}g}"


def vec(a, k=8):
    a = list(a)
    return "[" + ", ".join(fv(v) for v in a[:k]) + (", …" if len(a) > k else "") + "]"


def ST(title, desc, result):
    """1 bước của thuật toán: tiêu đề, cách làm, kết quả của bước."""
    return dict(title=title, desc=desc, result=result)


def S(label, value, hint=""):
    return dict(label=label, value=value, hint=hint)


def data_series(xs, ys):
    x, y = thin(xs, ys)
    return dict(label="Dữ liệu f(x)", kind="points", x=x, y=y)


def node_note(nd):
    if nd.get("info"):
        return nd["info"]
    return (f"Dữ liệu rời rạc nên mỗi mốc lý thuyết được thay bằng điểm dữ liệu gần nhất "
            f"(lệch tối đa {nd['dev']:.3g}).") if nd["dev"] > 0 else None


# ---------------------------------------------------------------- Lagrange
@register("lagrange", "Nội suy Lagrange",
          "Đa thức nội suy P_n(x) = Σ yᵢ·Lᵢ(x), kèm sai số thực và ước lượng cận sai số R_n.",
          [P("n", "Bậc n (có n+1 mốc)", "int", 5, "Bỏ qua nếu chọn 'mọi điểm dữ liệu'"),
           P("node_type", "Cách chọn mốc", "select", "equal", options=NODE_OPTS),
           P("x_eval", "Điểm cần tính P(x)", "text", "", "Ngăn cách bằng ; — để trống = trung điểm [a,b]. Điểm đầu tiên là x̄ khi chọn mốc gần nhất")])
def run_lagrange(x, y, a, b, p):
    xs, ys = subset(x, y, a, b)
    q = parse_points(p["x_eval"], a, b)
    f = fit(xs, ys, a, b, p["n"], p["node_type"], q[0])
    xn, yn = f["xn"], f["yn"]
    n = len(xn) - 1
    D = np.array([np.prod(xn[i] - np.delete(xn, i)) for i in range(n + 1)])
    q = parse_points(p["x_eval"], a, b)
    Pq = lagrange_eval(xn, yn, q)
    cL = [np.poly(np.delete(xn, i)) / D[i] for i in range(n + 1)]       # hệ số Lᵢ(x), luỹ thừa giảm dần
    cP = sum(yn[i] * cL[i] for i in range(n + 1))                       # hệ số P_n(x), luỹ thừa giảm dần
    pchk = float(np.max(np.abs(np.polyval(cP, xn) - yn)))
    fac = lambda v: f"(x {'−' if v >= 0 else '+'} {fv(abs(v))})"
    prodL = [" ".join(fac(xj) for jj, xj in enumerate(xn) if jj != i) + f" / {fv(D[i])}" for i in range(n + 1)]
    poly_tables = [
        dict(title="Đa thức nội suy P_n(x) dạng chính tắc: hệ số theo luỹ thừa giảm dần", columns=[f"x^{n - j}" for j in range(n + 1)], rows=[list(cP)]),
        dict(title="Các đa thức Lagrange cơ bản Lᵢ(x) khai triển (hàng i: hệ số theo luỹ thừa giảm dần)",
             columns=["i", "xᵢ", "yᵢ", "Dᵢ"] + [f"x^{n - j}" for j in range(n + 1)], rows=[[i, xn[i], yn[i], D[i]] + list(cL[i]) for i in range(n + 1)])]
    dm = dd_max(xs, ys, n + 1)
    rows = []
    for xv, pv in zip(q, Pq):
        j = int(np.argmin(np.abs(xs - xv)))
        hit = abs(xs[j] - xv) <= 1e-9 * max(1, abs(xv))
        rows.append([xv, pv, ys[j] if hit else None, abs(ys[j] - pv) if hit else None,
                     dm * float(w_abs(xn, xv)[0]) if dm is not None else None])
    L = [float(np.prod((q[0] - np.delete(xn, i)) / (xn[i] - np.delete(xn, i)))) for i in range(n + 1)]
    cs = np.cumsum(np.array(yn) * np.array(L))
    grid = f["grid"]
    notes = [node_note(f["nd"]), "P_n(x) = " + poly_str(cP[::-1]) + "   (dạng chính tắc)",
             "Cận sai số |R_n(x)| ≤ M_{n+1}/(n+1)! · |w_{n+1}(x)| với M_{n+1}/(n+1)! được ước lượng từ sai phân chia "
             "bậc n+1 của dữ liệu — chỉ là ước lượng, nhạy với nhiễu khi n lớn."]
    if np.any((q < a) | (q > b)):
        notes.append("Có điểm nằm ngoài [a, b]: đây là ngoại suy, sai số có thể rất lớn.")
    algo_steps = [
        ST("Chọn các mốc nội suy", f"Từ dữ liệu trong [a, b] chọn {n + 1} mốc ({NODE_NAME[p['node_type']]}).", f"xᵢ = {vec(xn)}\nyᵢ = {vec(yn)}"),
        ST("Tính Dᵢ = Π_{j≠i}(xᵢ − xⱼ)", "Mẫu số của đa thức Lagrange cơ bản Lᵢ.", f"Dᵢ = {vec(D)}"),
        ST("Viết từng Lᵢ(x) = Π_{j≠i}(x − xⱼ)/Dᵢ", "Tử số là tích các nhân tử (x − xⱼ) với j ≠ i.", "\n".join(f"L{i}(x) = {prodL[i]}" for i in range(min(n + 1, 8))) + ("\n…" if n > 7 else "")),
        ST("Lập đa thức nội suy dạng Lagrange", "P_n(x) = Σ yᵢ·Lᵢ(x).", "P_n(x) = " + " + ".join(f"{fv(yn[i])}·L{i}(x)" for i in range(min(n + 1, 8))) + (" + …" if n > 7 else "")),
        ST("Khai triển P_n(x) về dạng chính tắc", "Nhân các nhân tử và cộng các hệ số cùng bậc; kiểm tra P_n(xᵢ) = yᵢ.", "P_n(x) = " + poly_str(cP[::-1]) + f"\nmax |P_n(xᵢ) − yᵢ| = {fv(pchk)}"),
        ST(f"Tính Lᵢ(x̄) tại x̄ = {fv(q[0])}", "Lᵢ(x) = Π_{j≠i}(x − xⱼ) / Dᵢ.", f"Lᵢ(x̄) = {vec(L)}"),
        ST("Cộng P_n(x̄) = Σ yᵢ·Lᵢ(x̄)", "Cộng dần từng số hạng yᵢ·Lᵢ(x̄).", f"Tổng tích luỹ = {vec(cs)}\nP_n(x̄) = {fv(Pq[0])}"),
        ST("Đánh giá sai số", "So với giá trị thật trong dữ liệu; cận theo |R_n| ≤ M_{n+1}/(n+1)!·|w_{n+1}(x)|.",
           f"Sai số thực lớn nhất = {fv(f['max'])} (tại x = {fv(f['argmax'])})\nsup|R_n| ước lượng = {fv(dm * f['wsup']) if dm is not None else '—'}")]
    return dict(steps=algo_steps, 
        summary=[S("Bậc n", n), S("Số mốc", n + 1), S("max |P_n(xᵢ) − yᵢ|", pchk, "kiểm tra đa thức khai triển"), S("Kiểu mốc", NODE_NAME[p["node_type"]]),
                 S("Sai số thực lớn nhất", f["max"], f"tại x = {f['argmax']:.6g}"),
                 S("Sai số RMS", f["rmse"]), S("sup |w_{n+1}(x)|", f["wsup"]),
                 S("sup |R_n| ước lượng", None if dm is None else dm * f["wsup"], "cận trên theo công thức sai số")],
        tables=poly_tables + [dict(title="Giá trị P_n tại các điểm cần tính",
                     columns=["x", "P_n(x)", "f(x) trong dữ liệu", "|f − P_n|", "Cận |R_n(x)| ước lượng"], rows=rows),
                dict(title=f"Các mốc nội suy, Dᵢ = Π(xᵢ − xⱼ) và Lᵢ(x={q[0]:.6g})",
                     columns=["i", "xᵢ", "yᵢ", "Dᵢ", "Lᵢ(x)", "yᵢ·Lᵢ(x)", "Tổng tích luỹ"],
                     rows=[[i, xn[i], yn[i], D[i], L[i], yn[i] * L[i], cs[i]] for i in range(n + 1)])],
        plots=[dict(title="f(x) và đa thức nội suy P_n(x)", clamp=True,
                    series=[data_series(xs, ys), dict(label="P_n(x)", kind="line", x=grid, y=lagrange_eval(xn, yn, grid)),
                            dict(label="Mốc nội suy", kind="nodes", x=xn, y=yn)]),
               dict(title="Sai số thực |f(x) − P_n(x)| tại các điểm dữ liệu", yscale="log",
                    series=[dict(label="|f − P_n|", kind="line", x=thin(xs, f["err"])[0], y=thin(xs, f["err"])[1])])],
        notes=[s for s in notes if s])


# ---------------------------------------------------------------- Vandermonde
@register("vandermonde", "Giải hệ Vandermonde",
          "Tìm hệ số a₀…aₙ của P_n(x) = a₀ + a₁x + … + aₙxⁿ bằng cách giải hệ P_n(xᵢ)=yᵢ; kiểm tra định thức và độ ổn định.",
          [P("n", "Bậc n (có n+1 mốc)", "int", 5, "Nên ≤ 15 (hệ rất xấu điều kiện khi n lớn)"),
           P("node_type", "Cách chọn mốc", "select", "equal", options=NODE_OPTS[:2]),
           P("x_eval", "Điểm cần tính P(x)", "text", "", "Ngăn cách bằng ;")])
def run_vandermonde(x, y, a, b, p):
    if p["n"] > 30:
        raise ValueError("Với n > 30 hệ Vandermonde không còn ý nghĩa số học. Hãy dùng Lagrange.")
    xs, ys = subset(x, y, a, b)
    nd = select_nodes(xs, ys, p["n"], a, b, p["node_type"])
    xn, yn = nd["x"], nd["y"]
    n = len(xn) - 1
    V = np.vander(xn, increasing=True)
    coef = np.linalg.solve(V, yn)
    cond = float(np.linalg.cond(V))
    logdet = float(np.linalg.slogdet(V)[1])
    logprod = float(sum(np.log(abs(xn[i] - xn[j])) for i in range(n + 1) for j in range(i)))
    grid = np.linspace(a, b, 600)
    Pv = np.polynomial.polynomial.polyval(grid, coef)
    Pl = lagrange_eval(xn, yn, grid)
    q = parse_points(p["x_eval"], a, b)
    qv, ql = np.polynomial.polynomial.polyval(q, coef), lagrange_eval(xn, yn, q)
    notes = [node_note(nd), "Công thức: det V = Π_{i>j}(xᵢ − xⱼ) ≠ 0 ⇒ hệ có nghiệm duy nhất, tức đa thức nội suy tồn tại và duy nhất."]
    if cond > 1e10:
        notes.append("Số điều kiện rất lớn: các hệ số tìm được mất nhiều chữ số có nghĩa. Dạng Lagrange ổn định hơn nhiều.")
    algo_steps = [
        ST("Chọn các mốc nội suy", f"Chọn {n + 1} mốc ({NODE_NAME[p['node_type']]}).", f"xᵢ = {vec(xn)}\nyᵢ = {vec(yn)}"),
        ST("Lập hệ Σ aₖ·xᵢᵏ = yᵢ", f"Ma trận Vandermonde V cỡ {n + 1}×{n + 1}, hàng i là (1, xᵢ, xᵢ², …, xᵢⁿ).", f"Hàng đầu của V = {vec(V[0])}"),
        ST("Kiểm tra định thức det V = Π_{i>j}(xᵢ − xⱼ) ≠ 0", "Định thức khác 0 ⇒ hệ có nghiệm duy nhất.", f"ln|det V| = {fv(logdet)}\nln Π|xᵢ − xⱼ| = {fv(logprod)}"),
        ST("Giải V·a = y", "Dùng khử Gauss (numpy.linalg.solve); số điều kiện đo độ ổn định.", f"a = {vec(coef)}\ncond(V) = {fv(cond)}"),
        ST(f"Tính P_n(x̄) bằng Horner tại x̄ = {fv(q[0])}", "So với công thức Lagrange để thấy sai số làm tròn.", f"P_n(x̄) = {fv(qv[0])}\nLagrange: {fv(ql[0])}")]
    return dict(steps=algo_steps, 
        summary=[S("Bậc n", n), S("Số điều kiện cond(V)", cond), S("ln|det V|", logdet),
                 S("ln Π|xᵢ − xⱼ|", logprod, "phải bằng ln|det V|"),
                 S("max |P_Vandermonde − P_Lagrange|", float(np.max(np.abs(Pv - Pl))), "chênh lệch do sai số làm tròn")],
        tables=([dict(title="Hệ phương trình Σ aₖ·xᵢᵏ = yᵢ (ma trận Vandermonde và vế phải)", columns=["xᵢ"] + [f"xᵢ^{k}" for k in range(n + 1)] + ["yᵢ"],
                     rows=[[xn[i]] + list(V[i]) + [yn[i]] for i in range(n + 1)])] if n <= 11 else []) + [dict(title="Hệ số của đa thức", columns=["k", "aₖ"], rows=[[k, c] for k, c in enumerate(coef)]),
                dict(title="Giá trị tại các điểm", columns=["x", "P_n (Vandermonde)", "P_n (Lagrange)"],
                     rows=[[q[i], qv[i], ql[i]] for i in range(len(q))])],
        plots=[dict(title="f(x) và P_n(x) từ hệ Vandermonde", clamp=True,
                    series=[data_series(xs, ys), dict(label="P_n(x)", kind="line", x=grid, y=Pv),
                            dict(label="Mốc nội suy", kind="nodes", x=xn, y=yn)])],
        notes=[s for s in notes if s] + ["P_n(x) = " + poly_str(coef)])


# ---------------------------------------------------------------- Chebyshev
@register("chebyshev", "Tối ưu mốc nội suy (Chebyshev)",
          "Mốc là n+1 nghiệm của T_{n+1}(x) làm sup|w_{n+1}(x)| nhỏ nhất; so với mốc cách đều.",
          [P("n", "Bậc n (có n+1 mốc)", "int", 8)])
def run_chebyshev(x, y, a, b, p):
    xs, ys = subset(x, y, a, b)
    n = p["n"]
    t, xt = cheb_targets(a, b, n)
    o = np.argsort(xt)
    t, xt = t[o], xt[o]
    fc, fe = fit(xs, ys, a, b, n, "chebyshev"), fit(xs, ys, a, b, n, "equal")
    theory = ((b - a) / 2) ** (n + 1) / 2 ** n
    g = fc["grid"]
    algo_steps = [
        ST("Tính tᵢ = cos(π/(2(n+1)) + iπ/(n+1))", f"n + 1 = {n + 1} nghiệm của T_{{n+1}} trên [−1, 1].", f"tᵢ = {vec(t)}"),
        ST("Đổi biến về [a, b]: xᵢ = ((b−a)tᵢ + (b+a))/2", "Mốc lý thuyết trên đoạn nội suy.", f"xᵢ = {vec(xt)}"),
        ST("Lấy điểm dữ liệu gần mốc lý thuyết nhất", "Dữ liệu rời rạc nên thay mỗi mốc bằng điểm gần nhất.", f"xᵢ dùng = {vec(fc['xn'])}\nlệch lớn nhất = {fv(fc['nd']['dev'])}"),
        ST("Kiểm tra T_{n+1}(tᵢ) ≈ 0", "Dùng công thức truy hồi T_{k+1} = 2t·T_k − T_{k−1}.", f"max |T_{{n+1}}(tᵢ)| = {fv(float(np.max(np.abs(cheb_T(n + 1, t)))))}"),
        ST("So sánh sup|w_{n+1}(x)|", "Giá trị nhỏ nhất lý thuyết là ((b−a)/2)^{n+1}/2ⁿ.",
           f"Lý thuyết = {fv(theory)}\nMốc Chebyshev = {fv(fc['wsup'])}\nMốc cách đều = {fv(fe['wsup'])}\nSai số thực: Chebyshev {fv(fc['max'])} | cách đều {fv(fe['max'])}")]
    return dict(steps=algo_steps, 
        summary=[S("sup|w| lý thuyết nhỏ nhất", theory, "((b−a)/2)^{n+1} / 2^n"),
                 S("sup|w| mốc Chebyshev (dữ liệu)", fc["wsup"]), S("sup|w| mốc cách đều", fe["wsup"]),
                 S("Sai số thực – Chebyshev", fc["max"]), S("Sai số thực – cách đều", fe["max"])],
        tables=[dict(title="Mốc Chebyshev  xᵢ = ((b−a)tᵢ + (b+a))/2",
                     columns=["i", "tᵢ", "xᵢ lý thuyết", "xᵢ dùng (từ dữ liệu)", "lệch", "T_{n+1}(tᵢ) ≈ 0"],
                     rows=[[i, t[i], xt[i], fc["xn"][i], abs(fc["xn"][i] - xt[i]), cheb_T(n + 1, t[i])] for i in range(n + 1)])],
        plots=[dict(title="|w_{n+1}(x)| — Chebyshev so với cách đều", yscale="linear",
                    series=[dict(label="Chebyshev", kind="line", x=g, y=w_abs(fc["xn"], g)),
                            dict(label="Cách đều", kind="line", x=g, y=w_abs(fe["xn"], g))])],
        notes=[s for s in [node_note(fc["nd"])] if s])


# ---------------------------------------------------------------- So sánh (hiện tượng Runge)
@register("compare_nodes", "So sánh mốc đều và Chebyshev",
          "Sai số nội suy theo n với hai cách chọn mốc — minh hoạ hiện tượng Runge ở ví dụ 1/(25x²+1).",
          [P("n_list", "Các giá trị n", "text", "4; 8; 12; 16", "Ngăn cách bằng ;")])
def run_compare(x, y, a, b, p):
    xs, ys = subset(x, y, a, b)
    ns = [int(float(s)) for s in re.split(r"[;\s,]+", p["n_list"].strip()) if s]
    rows, ok, notes = [], [], []
    for n in ns:
        try:
            fe, fc = fit(xs, ys, a, b, n, "equal"), fit(xs, ys, a, b, n, "chebyshev")
            rows.append([n, fe["max"], fc["max"], fe["wsup"], fc["wsup"]])
            ok.append((n, fe, fc))
        except ValueError as e:
            rows.append([n, None, None, None, None])
            notes.append(f"n = {n}: {e}")
    if not ok:
        raise ValueError("Không có n nào chạy được. " + " ".join(notes))
    n, fe, fc = ok[-1]
    g = fe["grid"]
    algo_steps = [ST(f"n = {r_[0]}", "Chọn n+1 mốc cách đều và mốc Chebyshev, nội suy, đo sai số lớn nhất trên dữ liệu.",
                     "không chạy được (dữ liệu quá thưa hoặc không đủ điểm)" if r_[1] is None else
                     f"Cách đều: sai số = {fv(r_[1])}, sup|w| = {fv(r_[3])}\nChebyshev: sai số = {fv(r_[2])}, sup|w| = {fv(r_[4])}") for r_ in rows]
    return dict(steps=algo_steps, 
        summary=[S("Số giá trị n", len(ns))],
        tables=[dict(title="Sai số thực lớn nhất trên dữ liệu",
                     columns=["n", "Sai số – cách đều", "Sai số – Chebyshev", "sup|w| – cách đều", "sup|w| – Chebyshev"], rows=rows)],
        plots=[dict(title="Sai số lớn nhất theo n", yscale="log",
                    series=[dict(label="Cách đều", kind="line", x=[r[0] for r in rows], y=[r[1] for r in rows]),
                            dict(label="Chebyshev", kind="line", x=[r[0] for r in rows], y=[r[2] for r in rows])]),
               dict(title=f"Đường nội suy với n = {n}", clamp=True,
                    series=[data_series(xs, ys), dict(label="Cách đều", kind="line", x=g, y=lagrange_eval(fe["xn"], fe["yn"], g)),
                            dict(label="Chebyshev", kind="line", x=g, y=lagrange_eval(fc["xn"], fc["yn"], g))])],
        notes=notes)
