# Phòng thí nghiệm phương pháp số (Flask)

## Chạy
```
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python app.py                   # mở http://127.0.0.1:5000
```
Trong Visual Studio / VS Code: mở thư mục này, chọn interpreter `.venv`, chạy `app.py`.
Giao diện dùng Chart.js từ CDN nên cần internet khi mở trang.

## File dữ liệu
xlsx, cột A là x, cột B là f(x). Dòng tiêu đề được bỏ qua tự động; chấp nhận `1,1` và `1.1`.
Nhập [a, b] ở giao diện — chỉ các điểm trong [a, b] được dùng. Mốc lý thuyết (cách đều, Chebyshev)
được thay bằng điểm dữ liệu gần nhất, nên dữ liệu càng dày kết quả càng sát lý thuyết.

## Thêm thuật toán mới
1. Tạo `numerics/ten_file.py`, viết hàm `fn(x, y, a, b, p)` và gắn `@register(id, tên, mô tả, [P(...), ...])`
   (xem `numerics/interpolation.py`). Hàm trả về dict: `summary`, `tables`, `plots`, `notes`.
2. Thêm `from . import ten_file` ở cuối `numerics/__init__.py`.
3. Tải lại trang — form tham số và hiển thị kết quả tự có, không cần sửa HTML/JS.
Hàm dùng chung (chọn mốc, Lagrange, sai phân chia…) nằm trong `numerics/core.py`.

## Newton tiến/lùi — trích xuất mốc
Chọn 'Trích xuất k = n+1 điểm gần x̄ nhất' (k = n+1), nhập x̄ ở ô 'Điểm cần tính' (điểm đầu tiên là x̄). Kết quả có thêm f′(x̄) ≈ P_n′(x̄).

## Nội suy trung tâm
Stirling (n chẵn, số mốc lẻ), Bessel (n lẻ, số mốc chẵn). Cần dữ liệu cách đều trong [a, b]; mốc được trích xuất quanh x̄ (điểm đầu tiên ở ô "Điểm cần tính").

## Nội suy ngược
Giải f(x) = ȳ: tìm khoảng cách ly nghiệm, rồi dùng hàm ngược (mốc bất kỳ) hoặc phương pháp lặp Newton tiến/lùi (mốc cách đều). Hiện kết quả từng lần lặp.

## Spline và bình phương tối thiểu
Spline tuyến tính / cấp 2 (cho m₀) / cấp 3 (tự nhiên hoặc kẹp). Bình phương tối thiểu: đa thức bậc m, tổ hợp hàm cơ sở tuỳ ý (nhập biểu thức như x^2; sin(x)), mũ, luỹ thừa, ln.
