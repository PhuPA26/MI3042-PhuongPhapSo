const $ = s => document.querySelector(s);
let token = null, info = null, algos = [], last = null;

async function api(url, opts) {
  const r = await fetch(url, opts), j = await r.json();
  if (!r.ok) throw new Error(j.error || "Có lỗi xảy ra");
  return j;
}
function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v);
  const m = Math.abs(v);
  return m !== 0 && (m < 1e-4 || m >= 1e7) ? v.toExponential(5) : String(+v.toPrecision(8));
}
function showError(msg) { const e = $("#error"); e.hidden = !msg; e.textContent = msg || ""; }

// ---- dữ liệu
async function loaded(d) {
  token = d.token; info = d; showError("");
  $("#info").hidden = false;
  $("#info").innerHTML = `<b>${d.name}</b><br>${d.count} điểm · x ∈ [${fmt(d.xmin)}, ${fmt(d.xmax)}] · bước ≈ ${fmt(d.step)}${d.uniform ? " (đều)" : " (không đều)"}`
    + (d.skipped ? `<br>Bỏ qua ${d.skipped} dòng không phải số (tiêu đề?).` : "");
  $("#a").value = d.xmin; $("#b").value = d.xmax;
  $("#s-range").hidden = $("#s-algo").hidden = false;
  $("#empty").hidden = true;
}
$("#file").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  try { await loaded(await api("/api/upload", { method: "POST", body: fd })); } catch (err) { showError(err.message); $("#empty").hidden = true; }
};
document.querySelectorAll("[data-sample]").forEach(b => b.onclick = async () => {
  try { await loaded(await api("/api/sample/" + b.dataset.sample, { method: "POST" })); } catch (err) { showError(err.message); }
});

// ---- form tham số dựng từ registry
function buildParams() {
  const al = algos.find(a => a.id === $("#algo").value);
  $("#algo-desc").textContent = al.description;
  $("#params").innerHTML = al.params.map(p => {
    const id = "p_" + p.name, h = p.help ? `<span class="hint">${p.help}</span>` : "";
    if (p.type === "select") return `<label>${p.label}<select id="${id}">${p.options.map(o => `<option value="${o[0]}" ${o[0] === p.default ? "selected" : ""}>${o[1]}</option>`).join("")}</select>${h}</label>`;
    if (p.type === "checkbox") return `<label class="chk"><input type="checkbox" id="${id}" ${p.default ? "checked" : ""}> ${p.label}</label>`;
    return `<label>${p.label}<input type="text" id="${id}" value="${p.default ?? ""}">${h}</label>`;
  }).join("");
}
function readParams() {
  const al = algos.find(a => a.id === $("#algo").value), out = {};
  al.params.forEach(p => { const el = $("#p_" + p.name); out[p.name] = p.type === "checkbox" ? el.checked : el.value; });
  return out;
}
$("#algo").onchange = buildParams;

$("#run").onclick = async () => {
  const btn = $("#run"); btn.disabled = true; btn.textContent = "Đang tính…"; showError("");
  try {
    last = await api("/api/run", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, algorithm: $("#algo").value, a: $("#a").value, b: $("#b").value, params: readParams() }) });
    render();
  } catch (err) { showError(err.message); }
  btn.disabled = false; btn.textContent = "Tính";
};

// ---- hiển thị kết quả
const esc = t => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function render() {
  const r = last; if (!r) return;
  $("#out").hidden = false; $("#out-title").textContent = r.title;
  $("#summary").innerHTML = r.summary.map(s => `<div class="stat"><small>${s.label}</small><b>${fmt(s.value)}</b>${s.hint ? `<i>${s.hint}</i>` : ""}</div>`).join("");
  $("#steps").innerHTML = (r.steps || []).length
    ? `<h3 class="sec">Các bước của thuật toán và kết quả từng bước</h3><ol class="steps">${r.steps.map(s => `<li><b>${esc(s.title)}</b><p>${esc(s.desc)}</p><pre>${esc(s.result)}</pre></li>`).join("")}</ol>` : "";
  $("#notes").innerHTML = (r.notes || []).map(n => `<div class="note">${esc(n)}</div>`).join("");
  $("#tables").innerHTML = (r.tables || []).length ? `<h3 class="sec">Các bảng chi tiết</h3>` + r.tables.map(t => `<div class="tbl"><h3>${esc(t.title)}</h3><table><thead><tr>${t.columns.map(c => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${t.rows.map(row => `<tr>${row.map(v => `<td>${esc(fmt(v))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`).join("") : "";
}

(async () => {
  algos = await api("/api/algorithms");
  $("#algo").innerHTML = algos.map(a => `<option value="${a.id}">${a.name}</option>`).join("");
  buildParams();
})();
