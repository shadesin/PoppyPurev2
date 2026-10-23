// Browser verification using the installed Chrome's local DevTools connection.
// Start the app and an isolated Chrome on debugging port 9223 first.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";

const [photoPath, outputDirectory = "reports/generated/local-app"] = process.argv.slice(2);
if (!photoPath) throw new Error("Pass a path to a JPEG photo as the first argument.");
await fs.mkdir(outputDirectory, { recursive: true });
const target = await (await fetch("http://127.0.0.1:9223/json/new?about:blank", { method: "PUT" })).json();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let nextId = 0;
const pending = new Map();
const browserErrors = [];
socket.onmessage = ({ data }) => {
  const message = JSON.parse(data);
  if (message.method === "Runtime.exceptionThrown") browserErrors.push(message.params);
  if (message.id) {
    const request = pending.get(message.id);
    if (request) {
      clearTimeout(request.timer);
      pending.delete(message.id);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    }
  }
};
function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)); }, 20000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(expression) {
  return evaluate(`new Promise((resolve, reject) => {
    const started = Date.now();
    const check = () => {
      if (${expression}) return resolve(true);
      if (Date.now() - started > 15000) return reject(new Error('UI state timed out'));
      setTimeout(check, 50);
    }; check();
  })`);
}
async function screenshot(name) {
  const { data } = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await fs.writeFile(path.join(outputDirectory, name), Buffer.from(data, "base64"));
}
async function result() {
  await waitFor(`document.querySelector('#result-panel').getAttribute('aria-busy') === 'false' && document.querySelector('#percentage').textContent !== '—'`);
  return evaluate(`({ percentage: document.querySelector('#percentage').textContent,
    activeLevel: document.querySelector('[data-level].active')?.dataset.level,
    errorHidden: document.querySelector('#error').hidden,
    clearEnabled: !document.querySelector('#clear').disabled })`);
}

try {
  await send("Page.enable");
  await send("Runtime.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1200, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: "http://127.0.0.1:8765" });
  await waitFor(`document.querySelector('#drop-zone') && typeof selectFile === 'function'`);
  assert.equal(await evaluate(`document.querySelector('#percentage').textContent`), "—");
  await screenshot("desktop-empty.png");

  const document = await send("DOM.getDocument");
  const { nodeId } = await send("DOM.querySelector", { nodeId: document.root.nodeId, selector: "#file-input" });
  await send("DOM.setFileInputFiles", { nodeId, files: [path.resolve(photoPath)] });
  const selected = await result();
  assert.equal(selected.percentage, selected.activeLevel);
  assert.equal(selected.errorHidden, true);
  assert.equal(selected.clearEnabled, true);
  await screenshot("desktop-result.png");

  await evaluate(`document.querySelector('#clear').click()`);
  assert.equal(await evaluate(`document.querySelector('#preview').hidden && document.querySelector('#percentage').textContent === '—'`), true);

  const base64 = (await fs.readFile(photoPath)).toString("base64");
  await evaluate(`window.testPhoto = new File([Uint8Array.from(atob(${JSON.stringify(base64)}), c => c.charCodeAt(0))], 'sample.jpg', {type:'image/jpeg'});`);
  await evaluate(`{
    const transfer = new DataTransfer(); transfer.items.add(window.testPhoto);
    document.querySelector('#drop-zone').dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: transfer }));
  }`);
  const dropped = await result();
  assert.equal(dropped.percentage, selected.percentage);

  await evaluate(`{
    const transfer = new DataTransfer(); transfer.items.add(window.testPhoto);
    document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: transfer, bubbles: true, cancelable: true }));
  }`);
  const pasted = await result();
  assert.equal(pasted.percentage, selected.percentage);

  await evaluate(`selectFile(new File(['unsupported'], 'sample.heic', {type:'image/heic'}))`);
  assert.equal(await evaluate(`!document.querySelector('#error').hidden && document.querySelector('#error').textContent.includes('HEIC') && document.querySelector('#percentage').textContent === '—'`), true);

  await evaluate(`new Promise(resolve => {
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 200;
    canvas.toBlob(blob => { selectFile(new File([blob], 'small.png', {type:'image/png'})); resolve(); });
  })`);
  await waitFor(`document.querySelector('#result-panel').getAttribute('aria-busy') === 'false' && !document.querySelector('#error').hidden`);
  assert.equal(await evaluate(`document.querySelector('#error').textContent.includes('too small') && document.querySelector('#percentage').textContent === '—'`), true);

  await evaluate(`selectFile(window.testPhoto)`);
  const recovered = await result();
  assert.equal(recovered.percentage, selected.percentage);
  await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await evaluate(`document.querySelector('details').open = true`);
  assert.equal(await evaluate(`document.documentElement.scrollWidth <= window.innerWidth`), true);
  await screenshot("mobile-result.png");
  assert.deepEqual(browserErrors, []);
  const acceptance = {
    status: "passed", browser: "installed Google Chrome (headless)",
    checks: ["empty state", "file selection", "drag and drop", "image paste event", "clear", "invalid format clears previous result", "undersized image error", "error recovery", "390px responsive layout", "no JavaScript exceptions"],
    predicted_volume_percent: Number(selected.percentage),
    snapshots: ["desktop-empty.png", "desktop-result.png", "mobile-result.png"],
    note: "Clipboard events were simulated with an image file; the system clipboard was not accessed. Predictions are functional checks, not an accuracy evaluation."
  };
  await fs.writeFile(path.join(outputDirectory, "BROWSER_ACCEPTANCE.json"), JSON.stringify(acceptance, null, 2) + "\n");
  console.log(JSON.stringify(acceptance, null, 2));
} finally {
  await fetch(`http://127.0.0.1:9223/json/close/${target.id}`).catch(() => {});
  socket.close();
}
