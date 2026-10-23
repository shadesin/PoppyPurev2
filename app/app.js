"use strict";

const $ = (selector) => document.querySelector(selector);
const dropZone = $("#drop-zone");
const input = $("#file-input");
const preview = $("#preview");
const clear = $("#clear");
const errorBox = $("#error");
const retry = $("#retry");
let selectedFile = null;
let previewUrl = null;
let busy = false;

if (!/Mac|iPhone|iPad/.test(navigator.platform)) {
  $("#paste-shortcut").textContent = "Ctrl V";
}

function clearResult(message = "Waiting for your image") {
  $("#percentage").textContent = "—";
  $("#status").textContent = message;
  document.querySelectorAll("[data-level]").forEach((level) => {
    level.classList.remove("active");
    level.removeAttribute("aria-current");
  });
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  clear.disabled = value;
  retry.disabled = value;
  dropZone.setAttribute("aria-disabled", String(value));
  $("#result-panel").setAttribute("aria-busy", String(value));
}

function resetImage() {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  selectedFile = null;
  input.value = "";
  preview.removeAttribute("src");
  preview.hidden = true;
  $("#empty-preview").hidden = false;
  $("#replace-hint").hidden = true;
  $("#file-name").hidden = true;
  clear.hidden = true;
  retry.hidden = true;
  errorBox.hidden = true;
  clearResult();
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  clearResult("Could not analyse this image");
}

async function analyse() {
  if (!selectedFile || busy) return;
  errorBox.hidden = true;
  retry.hidden = true;
  clearResult("Analysing your image…");
  setBusy(true);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": selectedFile.type || "application/octet-stream" },
      body: selectedFile,
      signal: controller.signal,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed. Please retry.");
    const value = result.predicted_volume_percent;
    if (result.percentage_basis !== "volume" || ![0, 10, 20, 30, 40, 50, 100].includes(value)) {
      throw new Error("The app received an unexpected result. Please restart it.");
    }
    $("#percentage").textContent = String(value);
    $("#status").textContent = value === 0 ? "Classified as pure poppy" : value === 100 ? "Classified as semolina" : "Semolina detected";
    const level = document.querySelector(`[data-level="${value}"]`);
    level.classList.add("active");
    level.setAttribute("aria-current", "true");
  } catch (error) {
    const message = error.name === "AbortError"
      ? "This is taking longer than expected. Please retry or restart the app."
      : error instanceof TypeError
        ? "Cannot reach the local app. Check that its terminal is still running."
        : error.message;
    showError(message);
    retry.hidden = false;
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
}

function selectFile(file) {
  if (!file || busy) return;
  // Remove the previous prediction even when the new file is invalid.
  resetImage();
  const validType = ["image/jpeg", "image/png", "image/webp"].includes(file.type);
  const validExtension = !file.type && /\.(jpe?g|png|webp)$/i.test(file.name || "");
  if (!validType && !validExtension) {
    showError("Choose a JPEG, PNG or WebP photo. Export HEIC images as JPEG first.");
    return;
  }
  if (file.size === 0 || file.size > 20 * 1024 * 1024) {
    showError("Choose a non-empty image under 20 MB.");
    return;
  }
  selectedFile = file;
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.hidden = false;
  $("#empty-preview").hidden = true;
  $("#replace-hint").hidden = false;
  $("#file-name").textContent = `${file.name || "Pasted image"} · ${(file.size / 1024 / 1024).toFixed(1)} MB`;
  $("#file-name").hidden = false;
  clear.hidden = false;
  analyse();
}

dropZone.addEventListener("click", () => { if (!busy) input.click(); });
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (!busy) input.click();
  }
});
input.addEventListener("change", () => selectFile(input.files[0]));
clear.addEventListener("click", () => { if (!busy) resetImage(); });
retry.addEventListener("click", analyse);

let dragDepth = 0;
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropZone.classList.remove("dragging");
  selectFile(event.dataTransfer.files[0]);
});
dropZone.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth += 1;
  if (!busy) dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropZone.classList.remove("dragging");
});
document.addEventListener("paste", (event) => {
  const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.kind === "file" && entry.type.startsWith("image/"));
  if (item) {
    event.preventDefault();
    selectFile(item.getAsFile());
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !busy) resetImage();
});
window.addEventListener("pagehide", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
