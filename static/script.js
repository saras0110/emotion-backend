const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const emojiBubble = document.getElementById("emojiBubble");
const stateEl = document.getElementById("state");
const labelEl = document.getElementById("label");
const backendInput = document.getElementById("backendUrl");
const saveBtn = document.getElementById("saveUrl");

const LS_KEY = "emotion-backend-url";
let BACKEND_URL = localStorage.getItem(LS_KEY) || "";

backendInput.value = BACKEND_URL;
saveBtn.onclick = () => {
  BACKEND_URL = backendInput.value.trim();
  localStorage.setItem(LS_KEY, BACKEND_URL);
  stateEl.textContent = "Saved backend URL.";
};

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    stateEl.textContent = "Camera ready.";
  } catch (err) {
    stateEl.textContent = "Camera error: " + err.message;
  }
}

function drawFrame() {
  // mirror the video
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
  ctx.restore();
}

function captureBlob(quality = 0.85) {
  return new Promise(resolve => {
    canvas.toBlob(b => resolve(b), "image/jpeg", quality);
  });
}

let sending = false;
async function tick() {
  if (!BACKEND_URL) {
    stateEl.textContent = "Set your backend URL (Render) first.";
    requestAnimationFrame(tick);
    return;
  }
  drawFrame();

  if (!sending) {
    sending = true;
    try {
      const blob = await captureBlob(0.8);
      const form = new FormData();
      form.append("image", blob, "frame.jpg");

      const res = await fetch(`${BACKEND_URL.replace(/\/$/, "")}/predict`, {
        method: "POST",
        body: form
      });

      if (!res.ok) throw new Error("Backend HTTP " + res.status);
      const data = await res.json();

      if (data.success && data.face_found) {
        emojiBubble.textContent = data.emoji || "🙂";
        labelEl.textContent = `Emotion: ${data.label} (${(data.probabilities?.[data.label] * 100).toFixed(1)}%)`;

        // Optional: draw face bbox on canvas
        if (data.bbox) {
          const { x, y, w, h } = data.bbox;
          // Because we mirrored, we need to mirror the rectangle, too.
          ctx.save();
          ctx.scale(-1, 1);
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
          const xMirrored = canvas.width - (x + w);
          ctx.strokeRect(-xMirrored, y, w, h); // mirrored coords
          ctx.restore();
        }
      } else if (data.success && !data.face_found) {
        emojiBubble.textContent = "🔍";
        labelEl.textContent = "No face detected";
      } else {
        emojiBubble.textContent = "⚠️";
        labelEl.textContent = data.error || "Error";
      }
    } catch (err) {
      labelEl.textContent = "Error: " + err.message;
    } finally {
      sending = false;
    }
  }

  // ~4 FPS to keep bandwidth/CPU low
  setTimeout(() => requestAnimationFrame(tick), 250);
}

initCamera().then(() => requestAnimationFrame(tick));
