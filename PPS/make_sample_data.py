"""Tạo file xlsx mẫu (cột A: x, cột B: f(x)) để thử: python make_sample_data.py"""
import numpy as np
from openpyxl import Workbook

def save(name, x, f):
    wb = Workbook(); ws = wb.active; ws.append(["x", "f(x)"])
    for xi in x: ws.append([float(xi), float(f(xi))])
    wb.save(f"samples/{name}.xlsx")

save("runge_-1_1", np.round(np.linspace(-1, 1, 2001), 6), lambda x: 1 / (25 * x ** 2 + 1))
save("exp3_-2_2", np.round(np.linspace(-2, 2, 801), 6), lambda x: 3.0 ** x)
