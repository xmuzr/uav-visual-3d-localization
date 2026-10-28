"use strict";
const el = {};
const state = {photos: [], selected: null, image: null, observations: new Map(), loadId: 0};
const numberText = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : "-";
function status(id, text, style = "muted") {
  el[id].textContent = text;
  el[id].className = `pill ${style}`;
}
async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function invalidateResult() {
  for (const id of ["quality", "count", "lon", "lat", "h", "angle", "rmse", "max-residual", "enu", "warnings"]) {
    el[`result-${id}`].textContent = "-";
  }
  status("result-status", "待计算");
}
function renderPhotos() {
  el["photo-list"].replaceChildren();
  for (const photo of state.photos) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `photo-item${state.selected?.path === photo.path ? " active" : ""}`;
    button.textContent = `${state.observations.has(photo.path) ? "✓ " : ""}${photo.name}`;
    button.addEventListener("click", () => selectPhoto(photo));
    el["photo-list"].append(button);
  }
}
function renderObservations() {
  const observations = [...state.observations.values()];
  el["observation-list"].replaceChildren();
  for (const observation of observations) {
    const row = document.createElement("div");
    row.className = "observation-card";
    const text = document.createElement("div");
    text.textContent = `${observation.id} · U ${observation.pixel[0].toFixed(2)}, V ${observation.pixel[1].toFixed(2)}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "移除";
    remove.addEventListener("click", () => {
      state.observations.delete(observation.image_path);
      invalidateResult();
      renderObservations();
      renderPhotos();
    });
    row.append(text, remove);
    el["observation-list"].append(row);
  }
  status("ready-status", `${observations.length} 张已记录`, observations.length >= 2 ? "ready" : "muted");
  el["run-intersection"].disabled = observations.length < 2;
}
async function loadDirectory() {
  try {
    const result = await api(`/api/photos?${new URLSearchParams({directory: el["directory-input"].value.trim()})}`);
    state.loadId++;
    state.photos = result.photos;
    state.selected = null;
    state.image = null;
    state.observations.clear();
    el["photo-canvas"].width = el["photo-canvas"].height = 1;
    el["empty-state"].hidden = false;
    el["empty-state"].style.display = "grid";
    el["selected-name"].textContent = "未选择照片";
    el["image-size"].textContent = "0 x 0";
    el["pixel-u"].value = el["pixel-v"].value = "";
    el["directory-input"].value = result.directory;
    el["directory-status"].textContent = "已加载";
    el["photo-count"].textContent = `${result.photos.length} 张`;
    invalidateResult();
    renderPhotos();
    renderObservations();
  } catch (error) { el["directory-status"].textContent = error.message; }
}
async function selectPhoto(photo) {
  const loadId = ++state.loadId;
  state.selected = photo;
  state.image = null;
  el["record-point"].disabled = true;
  renderPhotos();
  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("照片加载失败"));
      image.src = `/api/image?${new URLSearchParams({path: photo.path})}`;
    });
    if (loadId !== state.loadId) return;
    state.image = image;
    el["selected-name"].textContent = photo.name;
    el["image-size"].textContent = `${image.naturalWidth} x ${image.naturalHeight}`;
    el["empty-state"].style.display = "none";
    const point = state.observations.get(photo.path)?.pixel || [image.naturalWidth / 2, image.naturalHeight / 2];
    el["pixel-u"].value = point[0];
    el["pixel-v"].value = point[1];
    el["record-point"].disabled = false;
    el["zoom-range"].value = 100;
    draw();
  } catch (error) { if (loadId === state.loadId) status("result-status", error.message, "warn"); }
}
function draw() {
  if (!state.image) return;
  const image = state.image, canvas = el["photo-canvas"];
  const fit = Math.min((el["viewer-scroll"].clientWidth - 50) / image.naturalWidth,
    (el["viewer-scroll"].clientHeight - 50) / image.naturalHeight, 1);
  const scale = Math.max(0.01, fit) * Number(el["zoom-range"].value) / 100;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.style.width = `${canvas.width * scale}px`;
  canvas.style.height = `${canvas.height * scale}px`;
  el["zoom-output"].textContent = `${el["zoom-range"].value}%`;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  const u = Number(el["pixel-u"].value), v = Number(el["pixel-v"].value);
  context.strokeStyle = "#ff3b30";
  context.lineWidth = Math.max(2, 2 / scale);
  const radius = 10 / scale;
  context.beginPath();
  context.moveTo(u - radius, v); context.lineTo(u + radius, v);
  context.moveTo(u, v - radius); context.lineTo(u, v + radius);
  context.stroke();
}
function pixelAt(event) {
  const rect = el["photo-canvas"].getBoundingClientRect();
  return [(event.clientX - rect.left) * state.image.naturalWidth / rect.width,
    (event.clientY - rect.top) * state.image.naturalHeight / rect.height];
}
function recordPoint() {
  if (!state.image || !state.selected) return;
  const pixel = [Number(el["pixel-u"].value), Number(el["pixel-v"].value)];
  if ([el["pixel-u"].value, el["pixel-v"].value].some(value => value.trim() === "") ||
      !pixel.every(Number.isFinite) || pixel[0] < 0 || pixel[1] < 0 ||
      pixel[0] >= state.image.naturalWidth || pixel[1] >= state.image.naturalHeight) {
    status("result-status", "像素超出图像范围", "warn"); return;
  }
  state.observations.set(state.selected.path, {id: state.selected.name, image_path: state.selected.path, pixel});
  invalidateResult(); renderPhotos(); renderObservations();
}
async function intersect() {
  el["run-intersection"].disabled = true;
  status("result-status", "计算中");
  try {
    const data = await api("/api/multi-view/localize", {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({observations: [...state.observations.values()],
        pose_correction: el["pose-correction"].checked ? "lrf_if_valid" : "none"})});
    const result = data.result;
    const fields = {quality: result.quality_level, count: result.ray_count,
      lon: numberText(result.geodetic.lon, 9), lat: numberText(result.geodetic.lat, 9),
      h: numberText(result.geodetic.h), angle: numberText(result.min_ray_angle_deg),
      rmse: numberText(result.residual_rmse_m), "max-residual": numberText(result.residual_max_m),
      enu: result.point_enu.map(value => numberText(value)).join(", "), warnings: result.warnings.join("；") || "无"};
    for (const [key, value] of Object.entries(fields)) el[`result-${key}`].textContent = value;
    status("result-status", "已完成", "ready");
  } catch (error) { invalidateResult(); status("result-status", error.message, "warn"); }
  finally { el["run-intersection"].disabled = state.observations.size < 2; }
}
window.addEventListener("DOMContentLoaded", () => {
  for (const node of document.querySelectorAll("[id]")) el[node.id] = node;
  el["directory-input"].value = document.body.dataset.defaultDirectory;
  el["load-directory"].addEventListener("click", loadDirectory);
  el["record-point"].disabled = true;
  el["record-point"].addEventListener("click", recordPoint);
  el["run-intersection"].addEventListener("click", intersect);
  el["pose-correction"].addEventListener("change", invalidateResult);
  el["clear-observations"].addEventListener("click", () => {
    state.observations.clear(); invalidateResult(); renderPhotos(); renderObservations();
  });
  el["photo-canvas"].addEventListener("click", event => {
    if (!state.image) return;
    const [u, v] = pixelAt(event);
    el["pixel-u"].value = u.toFixed(2); el["pixel-v"].value = v.toFixed(2); draw();
  });
  el["photo-canvas"].addEventListener("mousemove", event => {
    if (state.image) el["hover-pixel"].textContent = pixelAt(event).map(value => value.toFixed(1)).join(", ");
  });
  for (const id of ["pixel-u", "pixel-v", "zoom-range"]) el[id].addEventListener("input", draw);
  el["fit-image"].addEventListener("click", () => { el["zoom-range"].value = 100; draw(); });
  window.addEventListener("resize", draw);
  loadDirectory();
});
