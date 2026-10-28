const state = {
  mode: "single",
  photos: [],
  selectedPath: "",
  image: null,
  imageWidth: 0,
  imageHeight: 0,
  fitScale: 1,
  selectedPixel: null,
  multiSelectedPaths: new Set(),
  multiObservations: {},
  drag: {
    active: false,
    moved: false,
    startX: 0,
    startY: 0,
    scrollLeft: 0,
    scrollTop: 0,
  },
};

const el = {};

window.addEventListener("DOMContentLoaded", () => {
  for (const id of [
    "mode-single",
    "mode-multi",
    "directory-input",
    "load-directory",
    "directory-status",
    "photo-count",
    "photo-list",
    "selected-name",
    "image-size",
    "zoom-range",
    "zoom-output",
    "fit-image",
    "viewer-scroll",
    "photo-canvas",
    "empty-state",
    "pixel-u",
    "pixel-v",
    "hover-pixel",
    "localize-point",
    "record-multi-point",
    "gps-status",
    "camera-name",
    "focal-meta",
    "capture-time",
    "xmp-meta",
    "camera-lat",
    "camera-lon",
    "camera-h",
    "height-agl",
    "fx",
    "fy",
    "cx",
    "cy",
    "distortion",
    "pose-mode",
    "image-up-row",
    "image-up-azimuth",
    "heading",
    "tilt",
    "roll",
    "single-lrf-correction",
    "localize-status",
    "result-pixel",
    "result-lat",
    "result-lon",
    "result-h",
    "result-enu",
    "result-agl-lat",
    "result-agl-lon",
    "result-agl-h",
    "result-agl-enu",
    "result-lrf-lat",
    "result-lrf-lon",
    "result-lrf-h",
    "result-lrf-enu",
    "result-height-error",
    "result-diagnostics",
    "multi-lrf-correction",
    "multi-ready-status",
    "clear-multi-points",
    "run-multi-view",
    "multi-observation-list",
    "multi-result-status",
    "multi-result-quality",
    "multi-result-count",
    "multi-result-lat",
    "multi-result-lon",
    "multi-result-h",
    "multi-result-angle",
    "multi-result-rmse",
    "multi-result-max-residual",
    "multi-result-enu",
    "multi-result-warnings",
    "multi-photo-result-list",
  ]) {
    el[id] = document.getElementById(id);
  }

  el["directory-input"].value = document.body.dataset.defaultDirectory;
  el["mode-single"].addEventListener("click", () => setMode("single"));
  el["mode-multi"].addEventListener("click", () => setMode("multi"));
  el["load-directory"].addEventListener("click", loadDirectory);
  el["fit-image"].addEventListener("click", fitImage);
  el["zoom-range"].addEventListener("input", updateCanvasScale);
  el["pose-mode"].addEventListener("change", syncPoseFields);
  el["localize-point"].addEventListener("click", localizePoint);
  el["record-multi-point"].addEventListener("click", recordMultiPoint);
  el["clear-multi-points"].addEventListener("click", clearMultiPoints);
  el["run-multi-view"].addEventListener("click", runMultiView);
  el["viewer-scroll"].addEventListener("wheel", zoomWithWheel, { passive: false });
  el["photo-canvas"].addEventListener("mousedown", beginCanvasDrag);
  el["photo-canvas"].addEventListener("mousemove", showHoverPixel);
  el["photo-canvas"].addEventListener("mouseleave", () => {
    el["hover-pixel"].textContent = "悬停像素";
  });
  window.addEventListener("mousemove", dragCanvas);
  window.addEventListener("mouseup", endCanvasDrag);
  el["pixel-u"].addEventListener("change", selectManualPixel);
  el["pixel-v"].addEventListener("change", selectManualPixel);
  window.addEventListener("resize", () => {
    if (state.image) {
      computeFitScale();
      updateCanvasScale();
    }
  });
  syncPoseFields();
  setMode("single");
  loadDirectory();
});

function setMode(mode) {
  state.mode = mode;
  el["mode-single"].classList.toggle("active", mode === "single");
  el["mode-multi"].classList.toggle("active", mode === "multi");
  for (const panel of document.querySelectorAll(".single-mode-panel")) {
    panel.classList.toggle("hidden", mode !== "single");
  }
  for (const panel of document.querySelectorAll(".multi-mode-panel")) {
    panel.classList.toggle("hidden", mode !== "multi");
  }
  el["localize-point"].classList.toggle("hidden", mode !== "single");
  el["record-multi-point"].classList.toggle("hidden", mode !== "multi");
  if (mode === "multi") {
    setPill("localize-status", "多照片", "muted");
    restoreMultiPixelForCurrentPhoto();
  } else {
    clearMultiResultStatusOnly();
    drawCanvas();
  }
  renderPhotoList();
  renderMultiObservations();
}

async function loadDirectory() {
  setText("directory-status", "加载中");
  try {
    const response = await apiGet("/api/photos", {
      directory: el["directory-input"].value.trim(),
    });
    state.photos = response.photos;
    el["directory-input"].value = response.directory;
    setText("directory-status", shortPath(response.directory));
    setText("photo-count", `${response.photos.length} 张`);
    renderPhotoList();
  } catch (error) {
    setText("directory-status", error.message);
    setText("photo-count", "0");
    el["photo-list"].replaceChildren();
  }
}

function renderPhotoList() {
  el["photo-list"].replaceChildren();
  for (const photo of state.photos) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "photo-item";
    button.dataset.path = photo.path;
    button.title = photo.path;
    button.classList.toggle("active", photo.path === state.selectedPath);
    button.classList.toggle("multi-selected", state.mode === "multi" && state.multiSelectedPaths.has(photo.path));

    const row = document.createElement("span");
    row.className = state.mode === "multi" ? "photo-row" : "photo-row single";
    const check = document.createElement("span");
    check.className = "multi-check";
    check.textContent = state.mode === "multi" ? (state.multiSelectedPaths.has(photo.path) ? "☑" : "☐") : "";
    const copy = document.createElement("span");
    copy.className = "photo-copy";
    const name = document.createElement("strong");
    name.textContent = photo.name;
    const meta = document.createElement("small");
    const observation = state.multiObservations[photo.path];
    meta.textContent = state.mode === "multi" && observation
      ? `已点 ${formatNumber(observation.pixel[0], 1)}, ${formatNumber(observation.pixel[1], 1)}`
      : formatBytes(photo.size_bytes);
    copy.append(name, meta);
    row.append(check, copy);
    button.append(row);
    button.addEventListener("click", () => {
      if (state.mode === "multi") {
        if (state.multiSelectedPaths.has(photo.path)) {
          state.multiSelectedPaths.delete(photo.path);
        } else {
          state.multiSelectedPaths.add(photo.path);
        }
        clearMultiResult();
      }
      openPhoto(photo, button);
      renderPhotoList();
      renderMultiObservations();
    });
    el["photo-list"].append(button);
  }
}

async function openPhoto(photo, button) {
  state.selectedPath = photo.path;
  for (const item of el["photo-list"].querySelectorAll(".photo-item")) {
    item.classList.toggle("active", item === button);
  }
  setText("selected-name", photo.name);
  setPill("localize-status", "加载", "muted");
  try {
    const [metadata, image] = await Promise.all([
      apiGet("/api/metadata", { path: photo.path }),
      loadHtmlImage(`/api/image?path=${encodeURIComponent(photo.path)}`),
    ]);
    state.image = image;
    state.imageWidth = image.naturalWidth;
    state.imageHeight = image.naturalHeight;
    el["photo-canvas"].width = state.imageWidth;
    el["photo-canvas"].height = state.imageHeight;
    el["empty-state"].classList.add("hidden");
    fillMetadata(metadata);
    state.selectedPixel = null;
    setPixelInputs("", "");
    clearResult();
    restoreMultiPixelForCurrentPhoto();
    drawCanvas();
    fitImage();
    setPill("localize-status", "待定位", "muted");
    if (state.mode === "multi") {
      setPill("localize-status", "记录同名点", "muted");
    }
  } catch (error) {
    setPill("localize-status", "失败", "warn");
    setText("selected-name", error.message);
  }
}

function fillMetadata(metadata) {
  const width = metadata.image.width;
  const height = metadata.image.height;
  setText("image-size", `${width} x ${height}`);
  setText("camera-name", [metadata.camera.make, metadata.camera.model].filter(Boolean).join(" ") || "-");
  setText(
    "focal-meta",
    metadata.camera.focal_length_35mm
      ? `${formatNumber(metadata.camera.focal_length_mm, 2)} mm / ${formatNumber(metadata.camera.focal_length_35mm, 1)} mm`
      : metadata.camera.focal_length_mm
        ? `${formatNumber(metadata.camera.focal_length_mm, 2)} mm`
        : "-"
  );
  setText("capture-time", metadata.capture_time || "-");
  setText("xmp-meta", formatXmp(metadata.xmp_candidates));

  el["cx"].value = formatInput(width * 0.5);
  el["cy"].value = formatInput(height * 0.5);
  if (metadata.camera.calibrated_focal_length_px) {
    el["fx"].value = formatInput(metadata.camera.calibrated_focal_length_px);
    el["fy"].value = formatInput(metadata.camera.calibrated_focal_length_px);
    el["cx"].value = formatInput(metadata.camera.calibrated_optical_center_x ?? width * 0.5);
    el["cy"].value = formatInput(metadata.camera.calibrated_optical_center_y ?? height * 0.5);
  } else if (metadata.camera.fx_estimate_from_35mm) {
    el["fx"].value = formatInput(metadata.camera.fx_estimate_from_35mm);
    el["fy"].value = formatInput(metadata.camera.fx_estimate_from_35mm);
  } else {
    el["fx"].value = "";
    el["fy"].value = "";
  }

  if (metadata.gps) {
    el["camera-lat"].value = metadata.gps.lat;
    el["camera-lon"].value = metadata.gps.lon;
    el["camera-h"].value = metadata.gps.h ?? 0;
    setPill("gps-status", "GPS 已读", "ready");
  } else {
    el["camera-lat"].value = "";
    el["camera-lon"].value = "";
    el["camera-h"].value = 0;
    setPill("gps-status", "无 GPS", "warn");
  }

  const xmp = metadata.xmp_candidates || {};
  if (isFiniteNumber(xmp.relative_altitude_m) && xmp.relative_altitude_m > 0) {
    el["height-agl"].value = formatInput(xmp.relative_altitude_m);
  }
  if (isFiniteNumber(xmp.gimbal_yaw_deg)) {
    el["image-up-azimuth"].value = formatInput(xmp.gimbal_yaw_deg);
    el["heading"].value = formatInput(xmp.gimbal_yaw_deg);
  }
  if (isFiniteNumber(xmp.gimbal_pitch_deg) && xmp.gimbal_pitch_deg >= -90 && xmp.gimbal_pitch_deg < 0) {
    const tilt = Math.max(0, Math.min(89.9, 90 + xmp.gimbal_pitch_deg));
    el["pose-mode"].value = "heading_tilt";
    el["tilt"].value = formatInput(tilt);
    if (isFiniteNumber(xmp.gimbal_roll_deg)) {
      el["roll"].value = formatInput(xmp.gimbal_roll_deg);
    }
  }
  syncPoseFields();
}

function formatXmp(xmp) {
  if (!xmp || Object.keys(xmp).length === 0) {
    return "-";
  }
  const values = [];
  if (isFiniteNumber(xmp.relative_altitude_m)) {
    values.push(`AGL ${formatNumber(xmp.relative_altitude_m, 2)}`);
  }
  if (isFiniteNumber(xmp.gimbal_pitch_deg)) {
    values.push(`P ${formatNumber(xmp.gimbal_pitch_deg, 2)}`);
  }
  if (isFiniteNumber(xmp.gimbal_yaw_deg)) {
    values.push(`Y ${formatNumber(xmp.gimbal_yaw_deg, 2)}`);
  }
  if (isFiniteNumber(xmp.calibrated_focal_length_px)) {
    values.push(`fx ${formatNumber(xmp.calibrated_focal_length_px, 2)}`);
  }
  return values.join(" | ") || "-";
}

function fitImage() {
  el["zoom-range"].value = "100";
  computeFitScale();
  updateCanvasScale();
}

function computeFitScale() {
  if (!state.image) {
    state.fitScale = 1;
    return;
  }
  const viewer = el["viewer-scroll"];
  const maxWidth = Math.max(180, viewer.clientWidth - 72);
  const maxHeight = Math.max(180, viewer.clientHeight - 72);
  state.fitScale = Math.min(maxWidth / state.imageWidth, maxHeight / state.imageHeight, 1);
}

function updateCanvasScale() {
  if (!state.image) {
    return;
  }
  const zoom = Number(el["zoom-range"].value) / 100;
  const scale = state.fitScale * zoom;
  el["photo-canvas"].style.width = `${Math.max(1, state.imageWidth * scale)}px`;
  el["photo-canvas"].style.height = `${Math.max(1, state.imageHeight * scale)}px`;
  el["zoom-output"].value = `${el["zoom-range"].value}%`;
  el["zoom-output"].textContent = `${el["zoom-range"].value}%`;
}

function zoomWithWheel(event) {
  if (!state.image) {
    return;
  }
  event.preventDefault();
  const pixelBeforeZoom = eventToPixel(event);
  const currentZoom = Number(el["zoom-range"].value);
  const zoomFactor = Math.exp(-event.deltaY * 0.0014);
  const nextZoom = clamp(
    currentZoom * zoomFactor,
    Number(el["zoom-range"].min),
    Number(el["zoom-range"].max)
  );
  if (Math.abs(nextZoom - currentZoom) < 0.01) {
    return;
  }

  el["zoom-range"].value = String(nextZoom);
  updateCanvasScale();

  const rect = el["photo-canvas"].getBoundingClientRect();
  const pointClientX = rect.left + (pixelBeforeZoom[0] / state.imageWidth) * rect.width;
  const pointClientY = rect.top + (pixelBeforeZoom[1] / state.imageHeight) * rect.height;
  el["viewer-scroll"].scrollLeft += pointClientX - event.clientX;
  el["viewer-scroll"].scrollTop += pointClientY - event.clientY;
}

function drawCanvas() {
  if (!state.image) {
    return;
  }
  const canvas = el["photo-canvas"];
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.image, 0, 0);
  if (state.selectedPixel) {
    drawCrosshair(ctx, state.selectedPixel[0], state.selectedPixel[1]);
  }
}

function drawCrosshair(ctx, u, v) {
  const arm = Math.max(16, Math.min(state.imageWidth, state.imageHeight) * 0.012);
  ctx.save();
  ctx.lineWidth = Math.max(2, Math.min(state.imageWidth, state.imageHeight) / 1500);
  ctx.strokeStyle = "#ff3d2e";
  ctx.beginPath();
  ctx.moveTo(u - arm, v);
  ctx.lineTo(u + arm, v);
  ctx.moveTo(u, v - arm);
  ctx.lineTo(u, v + arm);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(u, v, Math.max(5, arm * 0.22), 0, Math.PI * 2);
  ctx.strokeStyle = "#ffffff";
  ctx.stroke();
  ctx.restore();
}

function beginCanvasDrag(event) {
  if (!state.image || event.button !== 0) {
    return;
  }
  event.preventDefault();
  state.drag.active = true;
  state.drag.moved = false;
  state.drag.startX = event.clientX;
  state.drag.startY = event.clientY;
  state.drag.scrollLeft = el["viewer-scroll"].scrollLeft;
  state.drag.scrollTop = el["viewer-scroll"].scrollTop;
  el["viewer-scroll"].classList.add("dragging");
}

function dragCanvas(event) {
  if (!state.drag.active) {
    return;
  }
  event.preventDefault();
  const dx = event.clientX - state.drag.startX;
  const dy = event.clientY - state.drag.startY;
  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
    state.drag.moved = true;
  }
  el["viewer-scroll"].scrollLeft = state.drag.scrollLeft - dx;
  el["viewer-scroll"].scrollTop = state.drag.scrollTop - dy;
}

function endCanvasDrag(event) {
  if (!state.drag.active) {
    return;
  }
  const didMove = state.drag.moved;
  state.drag.active = false;
  el["viewer-scroll"].classList.remove("dragging");
  if (!didMove && pointInsideCanvas(event.clientX, event.clientY)) {
    selectCanvasPixel(event);
  }
}

function selectCanvasPixel(event) {
  if (!state.image) {
    return;
  }
  const [u, v] = eventToPixel(event);
  setSelectedPixel(u, v);
}

function showHoverPixel(event) {
  if (!state.image || state.drag.active) {
    return;
  }
  const [u, v] = eventToPixel(event);
  setText("hover-pixel", `悬停 ${formatNumber(u, 2)}, ${formatNumber(v, 2)}`);
}

function eventToPixel(event) {
  const rect = el["photo-canvas"].getBoundingClientRect();
  const u = clamp((event.clientX - rect.left) * state.imageWidth / rect.width, 0, state.imageWidth - 1);
  const v = clamp((event.clientY - rect.top) * state.imageHeight / rect.height, 0, state.imageHeight - 1);
  return [u, v];
}

function pointInsideCanvas(clientX, clientY) {
  const rect = el["photo-canvas"].getBoundingClientRect();
  return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
}

function selectManualPixel() {
  if (!state.image) {
    return;
  }
  const u = Number(el["pixel-u"].value);
  const v = Number(el["pixel-v"].value);
  if (Number.isFinite(u) && Number.isFinite(v)) {
    setSelectedPixel(clamp(u, 0, state.imageWidth - 1), clamp(v, 0, state.imageHeight - 1));
  }
}

function setSelectedPixel(u, v) {
  state.selectedPixel = [u, v];
  setPixelInputs(formatInput(u), formatInput(v));
  drawCanvas();
  setText("result-pixel", `${formatNumber(u, 3)}, ${formatNumber(v, 3)}`);
}

function restoreMultiPixelForCurrentPhoto() {
  if (state.mode !== "multi" || !state.selectedPath) {
    return;
  }
  const observation = state.multiObservations[state.selectedPath];
  if (observation) {
    state.selectedPixel = observation.pixel;
    setPixelInputs(formatInput(observation.pixel[0]), formatInput(observation.pixel[1]));
  } else {
    state.selectedPixel = null;
    setPixelInputs("", "");
  }
  drawCanvas();
}

function setPixelInputs(u, v) {
  el["pixel-u"].value = u;
  el["pixel-v"].value = v;
}

function syncPoseFields() {
  const isHeading = el["pose-mode"].value === "heading_tilt";
  el["image-up-row"].classList.toggle("hidden", isHeading);
  for (const row of document.querySelectorAll(".heading-row")) {
    row.classList.toggle("hidden", !isHeading);
  }
}

async function localizePoint() {
  if (!state.selectedPixel) {
    setPill("localize-status", "选点", "warn");
    return;
  }
  setPill("localize-status", "计算", "muted");
  try {
    const payload = {
      image_path: state.selectedPath,
      pixel: state.selectedPixel,
      camera: {
        lat: valueNumber("camera-lat"),
        lon: valueNumber("camera-lon"),
        h: valueNumber("camera-h", 0),
      },
      ground: {
        camera_height_agl_m: valueNumber("height-agl"),
      },
      intrinsics: {
        fx: valueNumber("fx"),
        fy: valueNumber("fy"),
        cx: valueNumber("cx"),
        cy: valueNumber("cy"),
        distortion: distortionValues(),
      },
      pose: posePayload(),
      pose_correction: el["single-lrf-correction"].checked ? "lrf_if_valid" : "none",
    };
    const result = await apiPost("/api/localize", payload);
    showResult(result);
    setPill("localize-status", "完成", "ready");
  } catch (error) {
    setPill("localize-status", error.message, "warn");
  }
}

function recordMultiPoint() {
  if (state.mode !== "multi") {
    return;
  }
  if (!state.selectedPath || !state.selectedPixel) {
    setPill("multi-ready-status", "先选点", "warn");
    return;
  }
  const photo = state.photos.find((item) => item.path === state.selectedPath);
  state.multiSelectedPaths.add(state.selectedPath);
  state.multiObservations[state.selectedPath] = {
    id: photo?.name || state.selectedPath,
    name: photo?.name || state.selectedPath,
    image_path: state.selectedPath,
    pixel: [state.selectedPixel[0], state.selectedPixel[1]],
  };
  clearMultiResult();
  renderPhotoList();
  renderMultiObservations();
  setPill("multi-ready-status", "已记录", "ready");
}

function clearMultiPoints() {
  state.multiSelectedPaths.clear();
  state.multiObservations = {};
  clearMultiResult();
  renderPhotoList();
  renderMultiObservations();
  restoreMultiPixelForCurrentPhoto();
}

async function runMultiView() {
  const observations = selectedMultiObservations();
  if (observations.length < 2) {
    setPill("multi-result-status", "至少两张", "warn");
    return;
  }
  setPill("multi-result-status", "计算", "muted");
  try {
    const result = await apiPost("/api/multi-view/localize", {
      pose_correction: el["multi-lrf-correction"].checked ? "lrf_if_valid" : "none",
      intrinsics: {
        distortion: distortionValues(),
      },
      observations,
    });
    showMultiResult(result);
    setPill("multi-result-status", "完成", "ready");
  } catch (error) {
    setPill("multi-result-status", error.message, "warn");
  }
}

function selectedMultiObservations() {
  const selected = [];
  for (const path of state.multiSelectedPaths) {
    const observation = state.multiObservations[path];
    if (observation) {
      selected.push({
        id: observation.name,
        image_path: observation.image_path,
        pixel: observation.pixel,
      });
    }
  }
  return selected;
}

function renderMultiObservations() {
  if (!el["multi-observation-list"]) {
    return;
  }
  el["multi-observation-list"].replaceChildren();
  const selected = [...state.multiSelectedPaths];
  for (const path of selected) {
    const photo = state.photos.find((item) => item.path === path);
    const observation = state.multiObservations[path];
    const row = document.createElement("div");
    row.className = `multi-observation ${observation ? "ready" : ""}`;
    const title = document.createElement("strong");
    title.textContent = photo?.name || path;
    const meta = document.createElement("small");
    meta.textContent = observation
      ? `已点 ${formatNumber(observation.pixel[0], 2)}, ${formatNumber(observation.pixel[1], 2)}`
      : "未点";
    row.append(title, meta);
    el["multi-observation-list"].append(row);
  }
  if (selected.length === 0) {
    const empty = document.createElement("div");
    empty.className = "multi-observation";
    empty.textContent = "在左侧照片列表中选择照片";
    el["multi-observation-list"].append(empty);
  }
  const readyCount = selectedMultiObservations().length;
  setPill("multi-ready-status", readyCount >= 2 ? `${readyCount} 张就绪` : `${readyCount} 张`, readyCount >= 2 ? "ready" : "muted");
}

function showMultiResult(payload) {
  const result = payload.result || {};
  const geo = result.geodetic || {};
  setText("multi-result-quality", result.quality_level || "-");
  setText("multi-result-count", result.ray_count ?? "-");
  setText("multi-result-lat", formatNumber(geo.lat, 9));
  setText("multi-result-lon", formatNumber(geo.lon, 9));
  setText("multi-result-h", `${formatNumber(geo.h, 3)} m`);
  setText("multi-result-angle", `${formatNumber(result.min_ray_angle_deg, 3)}°`);
  setText("multi-result-rmse", `${formatNumber(result.residual_rmse_m, 3)} m`);
  setText("multi-result-max-residual", `${formatNumber(result.residual_max_m, 3)} m`);
  setText("multi-result-enu", Array.isArray(result.point_enu) ? result.point_enu.map((value) => formatNumber(value, 3)).join(", ") : "-");
  setText("multi-result-warnings", Array.isArray(result.warnings) && result.warnings.length ? result.warnings.join(" | ") : "-");
  renderMultiPhotoResults(payload.observations || []);
}

function clearMultiResultStatusOnly() {
  setPill("multi-result-status", "待计算", "muted");
}

function clearMultiResult() {
  clearMultiResultStatusOnly();
  for (const id of [
    "multi-result-quality",
    "multi-result-count",
    "multi-result-lat",
    "multi-result-lon",
    "multi-result-h",
    "multi-result-angle",
    "multi-result-rmse",
    "multi-result-max-residual",
    "multi-result-enu",
    "multi-result-warnings",
  ]) {
    setText(id, "-");
  }
  renderMultiPhotoResults([]);
}

function renderMultiPhotoResults(observations) {
  if (!el["multi-photo-result-list"]) {
    return;
  }
  el["multi-photo-result-list"].replaceChildren();
  if (!Array.isArray(observations) || observations.length === 0) {
    const empty = document.createElement("div");
    empty.className = "multi-photo-result empty";
    empty.textContent = "计算后显示每张照片的单张定位结果";
    el["multi-photo-result-list"].append(empty);
    return;
  }
  for (const observation of observations) {
    const single = observation.single_photo || {};
    const card = document.createElement("article");
    card.className = "multi-photo-result";

    const title = document.createElement("strong");
    title.textContent = observation.id || observation.image_path || "photo";
    const meta = document.createElement("small");
    meta.textContent = Array.isArray(observation.pixel)
      ? `像素 ${formatNumber(observation.pixel[0], 2)}, ${formatNumber(observation.pixel[1], 2)}`
      : "像素 -";
    card.append(title, meta);

    const heightResults = single.height_results || {};
    appendSinglePhotoHeightModel(card, "AGL", heightResults.agl, single.primary_model === "agl");
    appendSinglePhotoHeightModel(card, "LRF", heightResults.lrf, single.primary_model === "lrf");
    el["multi-photo-result-list"].append(card);
  }
}

function appendSinglePhotoHeightModel(parent, label, result, isPrimary) {
  const block = document.createElement("section");
  block.className = `multi-photo-model ${result && !result.error ? "ready" : "muted"}`;

  const heading = document.createElement("div");
  heading.className = "multi-photo-model-heading";
  const name = document.createElement("strong");
  name.textContent = label;
  heading.append(name);
  if (isPrimary) {
    const badge = document.createElement("span");
    badge.textContent = "主用";
    heading.append(badge);
  }
  block.append(heading);

  if (!result) {
    const empty = document.createElement("div");
    empty.className = "multi-photo-error";
    empty.textContent = `${label} 高度数据缺失`;
    block.append(empty);
    parent.append(block);
    return;
  }
  if (result.error) {
    const error = document.createElement("div");
    error.className = "multi-photo-error";
    error.textContent = result.error;
    block.append(error);
    parent.append(block);
    return;
  }

  const geo = result.geodetic || {};
  const grid = document.createElement("dl");
  grid.className = "multi-photo-grid";
  appendResultCell(grid, "经度", formatNumber(geo.lon, 9));
  appendResultCell(grid, "纬度", formatNumber(geo.lat, 9));
  appendResultCell(grid, "高程", `${formatNumber(geo.h, 3)} m`);
  if (Array.isArray(result.enu)) {
    appendResultCell(grid, "ENU", result.enu.map((value) => formatNumber(value, 3)).join(", "));
  }
  if (result.delta_to_intersection) {
    appendResultCell(grid, "距交会", `${formatNumber(result.delta_to_intersection.spatial_m, 3)} m`);
    appendResultCell(grid, "水平差", `${formatNumber(result.delta_to_intersection.horizontal_m, 3)} m`);
    appendResultCell(grid, "高差", `${formatNumber(result.delta_to_intersection.vertical_m, 3)} m`);
  }
  if (result.plane_source) {
    appendResultCell(grid, "高度源", result.plane_source);
  }
  block.append(grid);
  parent.append(block);
}

function appendResultCell(parent, label, value) {
  const item = document.createElement("div");
  const dt = document.createElement("dt");
  const dd = document.createElement("dd");
  dt.textContent = label;
  dd.textContent = value || "-";
  item.append(dt, dd);
  parent.append(item);
}

function posePayload() {
  if (el["pose-mode"].value === "heading_tilt") {
    return {
      mode: "heading_tilt",
      heading_deg: valueNumber("heading", 0),
      tilt_from_nadir_deg: valueNumber("tilt", 0),
      roll_deg: valueNumber("roll", 0),
    };
  }
  return {
    mode: "nadir",
    image_up_azimuth_deg: valueNumber("image-up-azimuth", 0),
    roll_deg: valueNumber("roll", 0),
  };
}

function distortionValues() {
  const values = el["distortion"].value.split(",").map((value) => Number(value.trim()));
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new Error("畸变参数");
  }
  return values;
}

function showResult(result) {
  setText("result-pixel", `${formatNumber(result.pixel[0], 3)}, ${formatNumber(result.pixel[1], 3)}`);
  setText("result-lat", formatNumber(result.geodetic.lat, 9));
  setText("result-lon", formatNumber(result.geodetic.lon, 9));
  setText("result-h", `${formatNumber(result.geodetic.h, 3)} m`);
  setText(
    "result-enu",
    result.enu.map((value) => formatNumber(value, 3)).join(", ")
  );
  showHeightResult("agl", result.height_results?.agl);
  showHeightResult("lrf", result.height_results?.lrf);
  setText("result-height-error", formatHeightComparison(result.height_comparison));
  setText("result-diagnostics", formatDiagnostics(result.diagnostics));
}

function clearResult() {
  setText("result-pixel", "-");
  setText("result-lat", "-");
  setText("result-lon", "-");
  setText("result-h", "-");
  setText("result-enu", "-");
  showHeightResult("agl", null);
  showHeightResult("lrf", null);
  setText("result-height-error", "-");
  setText("result-diagnostics", "-");
}

function showHeightResult(prefix, result) {
  const latId = `result-${prefix}-lat`;
  const lonId = `result-${prefix}-lon`;
  const hId = `result-${prefix}-h`;
  const enuId = `result-${prefix}-enu`;
  if (!result) {
    setText(latId, "-");
    setText(lonId, "-");
    setText(hId, "-");
    setText(enuId, "-");
    return;
  }
  if (result.error) {
    setText(latId, "error");
    setText(lonId, "error");
    setText(hId, result.error);
    setText(enuId, "-");
    return;
  }
  setText(latId, formatNumber(result.geodetic.lat, 9));
  setText(lonId, formatNumber(result.geodetic.lon, 9));
  setText(hId, `${formatNumber(result.geodetic.h, 3)} m`);
  setText(enuId, result.enu.map((value) => formatNumber(value, 3)).join(", "));
}

function formatHeightComparison(comparison) {
  if (!comparison) {
    return "-";
  }
  const values = [];
  if (isFiniteNumber(comparison.horizontal_m)) {
    values.push(`水平 ${formatNumber(comparison.horizontal_m, 3)} m`);
  }
  if (isFiniteNumber(comparison.vertical_m)) {
    values.push(`高差 ${formatNumber(comparison.vertical_m, 3)} m`);
  }
  if (isFiniteNumber(comparison.spatial_m)) {
    values.push(`三维 ${formatNumber(comparison.spatial_m, 3)} m`);
  }
  if (Array.isArray(comparison.lrf_minus_agl_enu)) {
    values.push(`ΔENU ${comparison.lrf_minus_agl_enu.map((value) => formatNumber(value, 3)).join(", ")}`);
  }
  return values.join(" | ") || "-";
}

function formatDiagnostics(diagnostics) {
  if (!diagnostics || Object.keys(diagnostics).length === 0) {
    return "-";
  }
  const values = [];
  if (diagnostics.quality_level) {
    values.push(`quality ${diagnostics.quality_level}`);
  }
  if (diagnostics.metadata_match_ok !== undefined) {
    values.push(`MD ${diagnostics.metadata_match_ok ? "ok" : "check"}`);
  }
  if (diagnostics.correction_mode) {
    values.push(`mode ${diagnostics.correction_mode}`);
  }
  if (diagnostics.plane_source) {
    values.push(`plane ${diagnostics.plane_source}`);
  }
  if (diagnostics.lrf_used !== undefined) {
    values.push(`LRF ${diagnostics.lrf_used ? "used" : "unused"}`);
  }
  if (diagnostics.lrf_distance_ok !== undefined) {
    values.push(`LRF gate ${diagnostics.lrf_distance_ok ? "ok" : "blocked"}`);
  }
  if (isFiniteNumber(diagnostics.raw_optical_axis_miss_m)) {
    values.push(`raw miss ${formatNumber(diagnostics.raw_optical_axis_miss_m, 3)} m`);
  }
  if (isFiniteNumber(diagnostics.lrf_distance_residual_m)) {
    values.push(`LRF residual ${formatNumber(diagnostics.lrf_distance_residual_m, 3)} m`);
  }
  if (isFiniteNumber(diagnostics.corrected_shift_from_raw_m)) {
    values.push(`shift ${formatNumber(diagnostics.corrected_shift_from_raw_m, 3)} m`);
  }
  if (isFiniteNumber(diagnostics.rtk_horizontal_std_m)) {
    values.push(`RTK h ${formatNumber(diagnostics.rtk_horizontal_std_m, 4)} m`);
  }
  if (isFiniteNumber(diagnostics.rtk_vertical_std_m)) {
    values.push(`RTK z ${formatNumber(diagnostics.rtk_vertical_std_m, 4)} m`);
  }
  if (isFiniteNumber(diagnostics.flight_speed_mps)) {
    values.push(`speed ${formatNumber(diagnostics.flight_speed_mps, 2)} m/s`);
  }
  if (isFiniteNumber(diagnostics.lens_focus_delta_steps)) {
    values.push(`lens delta ${formatNumber(diagnostics.lens_focus_delta_steps, 0)}`);
  }
  if (diagnostics.flight_gimbal_yaw_consistent !== undefined) {
    values.push(`att ${diagnostics.flight_gimbal_yaw_consistent ? "ok" : "check"}`);
  }
  for (const key of [
    "metadata_match_warning",
    "attitude_warning",
    "motion_warning",
    "lens_warning",
    "lrf_distance_warning",
  ]) {
    if (diagnostics[key]) {
      values.push(diagnostics[key]);
    }
  }
  if (diagnostics.warning) {
    values.push(diagnostics.warning);
  }
  return values.join(" | ") || "-";
}

async function apiGet(url, query) {
  const params = new URLSearchParams(query);
  const response = await fetch(`${url}?${params.toString()}`);
  return readApiResponse(response);
}

async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readApiResponse(response);
}

async function readApiResponse(response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function loadHtmlImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("照片加载失败"));
    image.src = url;
  });
}

function valueNumber(id, fallback) {
  const raw = el[id].value;
  if (raw === "" && fallback !== undefined) {
    return fallback;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${el[id].previousElementSibling?.textContent || id}`);
  }
  return value;
}

function setText(id, value) {
  if (!el[id]) {
    return;
  }
  el[id].textContent = value;
}

function setPill(id, text, tone) {
  if (!el[id]) {
    return;
  }
  el[id].textContent = text;
  el[id].className = `pill ${tone}`;
}

function shortPath(path) {
  if (path.length <= 28) {
    return path;
  }
  return `...${path.slice(-27)}`;
}

function formatBytes(value) {
  if (value > 1024 * 1024) {
    return `${formatNumber(value / (1024 * 1024), 1)} MB`;
  }
  return `${formatNumber(value / 1024, 1)} KB`;
}

function formatInput(value) {
  return Number(value).toFixed(4).replace(/\.?0+$/, "");
}

function formatNumber(value, digits) {
  if (!isFiniteNumber(value)) {
    return "-";
  }
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
