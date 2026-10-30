"""Registry thuật toán. Thêm thuật toán mới: tạo hàm + @register(...) trong 1 file của gói này,
rồi import file đó ở cuối __init__.py. Giao diện web tự dựng form từ danh sách tham số."""
import numpy as np

REGISTRY = {}


def P(name, label, type="number", default=None, help="", options=None):
    """Khai báo 1 tham số. type: number | int | text | select | checkbox"""
    return dict(name=name, label=label, type=type, default=default, help=help, options=options or [])


def register(id, name, description, params=()):
    def deco(fn):
        REGISTRY[id] = dict(id=id, name=name, description=description, params=list(params), fn=fn)
        return fn
    return deco


def clean(o):
    """Đưa kết quả về dạng JSON an toàn (numpy -> python, NaN/inf -> None)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


from . import interpolation  # noqa: E402,F401  (đăng ký các thuật toán nội suy)
from . import newton  # noqa: E402,F401
from . import central  # noqa: E402,F401
from . import inverse  # noqa: E402,F401
from . import spline, lsq  # noqa: E402,F401
