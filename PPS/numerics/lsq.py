"""Phương pháp bình phương tối thiểu (slide 19): Φ, M = ΦᵀΦ, b = Φᵀy, giải M·a = b."""
import ast
import numpy as np
from . import register, P
from .core import subset, parse_points, thin
from .interpolation import S, ST, vec, fv, data_series

FUN = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "ln": np.log, "sqrt": np.sqrt,
       "abs": np.abs, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "arctan": np.arctan}
BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
MAXR = 20


def compile_expr(text):
    """Biên dịch biểu thức của x (an toàn: chỉ + − * / ^, hàm sin cos exp ln sqrt…, hằng pi, e)."""
    try:
        tree = ast.parse(text.strip().replace("^", "**").replace(",", "."), mode="eval").body
    except SyntaxError:
        raise ValueError(f"Không đọc được hàm cơ sở '{text}'. Ví dụ: x^2; sin(x); exp(2*x)")

    def ev(n, x):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return np.full_like(x, float(n.value))
        if isinstance(n, ast.Name) and n.id in ("x", "pi", "e"):
            return x if n.id == "x" else np.full_like(x, np.pi if n.id == "pi" else np.e)
        if isinstance(n, ast.BinOp) and type(n.op) in BIN:
            return BIN[type(n.op)](ev(n.left, x), ev(n.right, x))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return -ev(n.operand, x) if isinstance(n.op, ast.USub) else ev(n.operand, x)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUN and len(n.args) == 1:
            return FUN[n.func.id](ev(n.args[0], x))
        raise ValueError(f"Biểu thức '{text}' có thành phần không được hỗ trợ.")
    ev(tree, np.array([1.0]))                                    # kiểm tra cú pháp sớm
    return lambda x: ev(tree, np.asarray(x, float))


def basis_from(p, model):
    if model == "poly":
        names = ["1", "x"] + [f"x^{j}" for j in range(2, p["degree"] + 1)]
        return names[:p["degree"] + 1], [(lambda j: (lambda x: x ** j))(j) for j in range(p["degree"] + 1)]
    names = [t.strip() for t in str(p["basis"]).replace("\n", ";").split(";") if t.strip()]
    if not names:
        raise ValueError("Hãy nhập ít nhất 1 hàm cơ sở, ví dụ: x^2; sin(x)")
    return names, [compile_expr(t) for t in names]


def run_lsq(x, y, a, b, p):
    xs, ys = subset(x, y, a, b)
    n, model, s = len(xs), p["model"], (p["y_shift"] or 0.0)
    Y = ys + s
    names, fns = basis_from(p, model)
    cols = [np.broadcast_to(f(xs), xs.shape).astype(float) for f in fns]
    sign = 1.0
    if model in ("exp", "power"):
        if np.any(Y == 0) or np.any(np.sign(Y) != np.sign(Y[0])):
            raise ValueError("Mô hình mũ/luỹ thừa cần y + dịch cùng dấu và khác 0 tại mọi điểm (slide: y>0 hoặc y<0).")
        sign = float(np.sign(Y[0]))
        T = np.log(np.abs(Y))
        if model == "power":
            if np.any(xs <= 0):
                raise ValueError("Mô hình y = a·x^b cần x > 0.")
            cols, names = [np.log(xs)], ["ln x"]
        cols, pnames = [np.ones(n)] + cols, ["ln|a|"] + [f"b{i + 1}" for i in range(len(cols))]
    elif model == "log":
        if np.max(Y) > 700:
            raise ValueError("Giá trị y quá lớn để lấy e^y.")
        T, pnames = np.exp(Y), [f"b{i + 1}" for i in range(len(cols))]
    else:
        T, pnames = Y, [f"a{i + 1}" for i in range(len(cols))]
    m = len(cols)
    if n < m:
        raise ValueError(f"Có {n} điểm nhưng {m} tham số: cần n ≥ m.")
    Phi = np.column_stack(cols)
    M, bvec = Phi.T @ Phi, Phi.T @ T                              # M = ΦᵀΦ, b = Φᵀy
    cond = float(np.linalg.cond(M))
    if not np.isfinite(cond) or cond > 1e15:
        raise ValueError("Ma trận M gần như suy biến (các hàm cơ sở phụ thuộc tuyến tính hoặc dữ liệu quá ít).")
    coef = np.linalg.solve(M, bvec)

    def model_fn(xq):
        xq = np.asarray(xq, float)
        cv = [np.broadcast_to(f(xq), xq.shape).astype(float) for f in fns] if model != "power" else [np.log(np.maximum(xq, 1e-300))]
        if model in ("exp", "power"):
            return sign * np.exp(coef[0] + sum(c * v for c, v in zip(coef[1:], cv))) - s
        lin = sum(c * v for c, v in zip(coef, cv))
        return (np.log(lin) if model == "log" else lin) - s
    with np.errstate(all="ignore"):
        g = model_fn(xs)
        r = ys - g
        ok = np.isfinite(g)
        nv = int(ok.sum())
        SS = float(np.sum(r[ok] ** 2))
        sst = float(np.sum((ys[ok] - ys[ok].mean()) ** 2)) if nv else 0.0
    if model in ("exp", "power"):
        params = [("a", sign * np.exp(coef[0]))] + [(f"b{i + 1}" + ("" if model == "power" else f"  [hệ số của {names[i]}]"), c) for i, c in enumerate(coef[1:])]
        f_str = "y = " + (f"{params[0][1]:.6g}·x^{coef[1]:.6g}" if model == "power" else
                          f"{params[0][1]:.6g}·exp(" + " + ".join(f"{c:.6g}·({nm})" for c, nm in zip(coef[1:], names)) + ")") + (f" − {s:g}" if s else "")
    else:
        params = [(f"{pnames[i]}  [hệ số của {names[i]}]", c) for i, c in enumerate(coef)]
        lin = " + ".join(f"{c:.6g}·({nm})" for c, nm in zip(coef, names))
        f_str = "y = " + (f"ln({lin})" if model == "log" else lin) + (f" − {s:g}" if s else "")
    q = parse_points(p["x_eval"], xs[0], xs[-1])
    with np.errstate(all="ignore"):
        gq = model_fn(q)
    tables = [
        dict(title="Ma trận Φ (các cột là φⱼ(xᵢ)) – " + (f"{MAXR} dòng đầu" if n > MAXR else "toàn bộ"),
             columns=["xᵢ", "yᵢ (đã biến đổi)"] + (["1"] + names if model in ("exp",) else names if model != "power" else ["1", "ln x"]),
             rows=[[xs[i], T[i]] + list(Phi[i]) for i in range(min(n, MAXR))]),
        dict(title="Hệ phương trình chuẩn M·a = b, với M = ΦᵀΦ, b = Φᵀy", columns=["", *pnames, "b"],
             rows=[[pnames[i], *M[i], bvec[i]] for i in range(m)]),
        dict(title="Nghiệm", columns=["tham số", "giá trị"], rows=[[nm, v] for nm, v in params]),
        dict(title="Sai lệch tại các điểm dữ liệu" + (f" ({MAXR} dòng đầu)" if n > MAXR else ""), columns=["xᵢ", "yᵢ", "g(xᵢ)", "yᵢ − g(xᵢ)"],
             rows=[[xs[i], ys[i], g[i], r[i]] for i in range(min(n, MAXR))]),
        dict(title="Giá trị hàm xấp xỉ tại các điểm cần tính", columns=["x", "g(x)"], rows=[[q[i], gq[i]] for i in range(len(q))])]
    grid = np.linspace(xs[0], xs[-1], 400)
    with np.errstate(all="ignore"):
        gg = model_fn(grid)
    plots = [dict(title="Dữ liệu và hàm xấp xỉ g(x)", clamp=False,
                  series=[data_series(xs, ys), dict(label="g(x)", kind="line", x=grid, y=gg)]),
             dict(title="Sai lệch yᵢ − g(xᵢ)", series=[dict(label="sai lệch", kind="points", x=thin(xs, r)[0], y=thin(xs, r)[1])])]
    notes = [f_str, "Phương pháp: S = Σ(g(xᵢ) − yᵢ)² → min ⇔ M·a = b (Bước 1 nhập dữ liệu, Bước 2 tính Φ, M, b, Bước 3 giải hệ)."]
    if model in ("exp", "power", "log"):
        notes.append("Hàm được đưa về dạng tuyến tính theo tham số (lấy ln hoặc e^y) rồi cực tiểu hoá sai số trên dữ liệu đã biến đổi; "
                     "σ bên dưới tính trên dữ liệu gốc nên có thể chưa là nhỏ nhất theo nghĩa gốc.")
    if nv < n:
        notes.append(f"CẢNH BÁO: g(x) không xác định tại {n - nv}/{n} điểm (đối số của ln ≤ 0 hoặc tràn số), nên σ chỉ tính trên {nv} điểm còn lại. "
                     "Mô hình này không phù hợp với dữ liệu, hãy đổi mô hình hoặc hàm cơ sở.")
    summary = [S("Số điểm n", n), S("Số tham số m", m), S("Sai số trung bình phương σₙ = √(Σr²/n)", float(np.sqrt(SS / nv)) if nv else None),
               S("S = Σ(yᵢ − g(xᵢ))²", SS), S("max |yᵢ − g(xᵢ)|", float(np.nanmax(np.abs(r))) if nv else None),
               S("R²", 1 - SS / sst if sst > 0 else None), S("cond(M)", cond)]
    trans = {"poly": "Không biến đổi: y ≈ Σ aⱼxʲ.", "custom": "Không biến đổi: y ≈ Σ aⱼφⱼ(x).", "exp": "Lấy ln|y| ⇒ ln|y| = ln|a| + Σ bⱼφⱼ(x).",
             "power": "Lấy ln|y| ⇒ ln|y| = ln|a| + b·ln x.", "log": "Lấy eʸ ⇒ eʸ = Σ bⱼφⱼ(x)."}[model]
    algo_steps = [
        ST("Nhập dữ liệu và chọn mô hình", f"{n} điểm trong [a, b]; {trans}" + (f" Dữ liệu đã cộng s = {fv(s)}." if s else ""), f"hàm cơ sở: {', '.join(names)}\nsố tham số m = {m}"),
        ST("Lập ma trận Φ (n×m)", "Cột j là (φⱼ(x₁), …, φⱼ(xₙ))ᵗ.", f"Φ cỡ {n}×{m}\nhàng 1 = {vec(Phi[0])}"),
        ST("Tính M = ΦᵗΦ và b = Φᵗy", "Hệ phương trình chuẩn từ ∂S/∂aᵢ = 0.", "M =\n" + "\n".join("  " + vec(M[i]) for i in range(m)) + f"\nb = {vec(bvec)}"),
        ST("Giải M·a = b", "Khử Gauss; cond(M) đo độ ổn định.", f"nghiệm = {vec(coef)}\ncond(M) = {fv(cond)}"),
        ST("Hàm xấp xỉ và sai số", "S = Σ(yᵢ − g(xᵢ))², σₙ = √(S/n).", f"{f_str}\nS = {fv(SS)}, σₙ = {fv(float(np.sqrt(SS / nv))) if nv else '—'}")]
    return dict(steps=algo_steps, summary=summary, tables=tables, plots=plots, notes=notes)


register("lsq", "Bình phương tối thiểu",
         "Tìm g = Σ aⱼφⱼ(x) (hoặc dạng mũ, luỹ thừa, ln đưa về tuyến tính theo tham số) làm sai số trung bình phương nhỏ nhất; hiện Φ, M, b và nghiệm.",
         [P("model", "Mô hình", "select", "poly", options=[
             ["poly", "Đa thức bậc m (bậc nhất, bậc hai, …)"], ["custom", "Tổ hợp tuyến tính a₁φ₁ + … + aₘφₘ"],
             ["exp", "y = a·e^(b₁φ₁+…+bₘφₘ)"], ["power", "y = a·x^b"], ["log", "y = ln(b₁φ₁+…+bₘφₘ)"]]),
          P("degree", "Bậc m (mô hình đa thức)", "int", 2, "1 = hàm bậc nhất, 2 = hàm bậc hai"),
          P("basis", "Các hàm cơ sở φⱼ(x)", "text", "x^2; sin(x)", "Cho mô hình tổ hợp / mũ / ln. Ngăn cách bằng ;  Dùng x, ^, sin, cos, exp, ln, sqrt. Mô hình mũ ví dụ: x"),
          P("y_shift", "Cộng y thêm hằng số s", "number", 0.0, "Ví dụ y = a·e^(bx) − 2 ⇒ nhập s = 2 (xấp xỉ y + 2)"),
          P("x_eval", "Điểm cần tính g(x)", "text", "", "Ngăn cách bằng ;")])(run_lsq)
