import io
import os
import math
import base64
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------- Config ----------
FER_ONNX_PATH = os.getenv("FER_ONNX_PATH", "models/emotion-ferplus-8.onnx")
HAAR_PATH = os.getenv("HAAR_PATH", "data/haarcascade_frontalface_default.xml")

# Allow your GitHub Pages site + local dev
FRONTEND_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "https://YOUR_GITHUB_USERNAME.github.io",  # replace with your GH Pages root
    "https://YOUR_GITHUB_USERNAME.github.io/YOUR_PAGES_REPO"  # or specific project page
]

EMOTIONS = [
    "neutral","happiness","surprise","sadness",
    "anger","disgust","fear","contempt"
]
EMOJI_MAP = {
    "neutral":"😐","happiness":"😄","surprise":"😲","sadness":"😢",
    "anger":"😠","disgust":"🤢","fear":"😨","contempt":"😒"
}

# ---------- App ----------
app = FastAPI(title="Emotion API (FER+ ONNX)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load models ----------
if not os.path.exists(FER_ONNX_PATH):
    raise FileNotFoundError(f"FER+ ONNX model not found at {FER_ONNX_PATH}")

if not os.path.exists(HAAR_PATH):
    raise FileNotFoundError(f"Haar cascade not found at {HAAR_PATH}")

ort_sess = ort.InferenceSession(FER_ONNX_PATH, providers=["CPUExecutionProvider"])
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def preprocess_face(gray_img, bbox):
    (x,y,w,h) = bbox
    face = gray_img[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face_resized = cv2.resize(face, (64, 64), interpolation=cv2.INTER_AREA)
    # FER+ expects 1x1x64x64 float32
    face_arr = face_resized.astype(np.float32) / 255.0
    face_arr = np.expand_dims(np.expand_dims(face_arr, 0), 0)
    return face_arr

@app.get("/")
def root():
    return {"status":"ok","message":"Emotion API running"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        img = np.array(pil)[:, :, ::-1]  # to BGR for OpenCV

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return JSONResponse({"success": True, "face_found": False, "message":"No face detected"}, status_code=200)

        # use largest face
        faces_sorted = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        bbox = tuple(map(int, faces_sorted[0]))

        x,y,w,h = bbox
        inp = preprocess_face(gray, bbox)
        if inp is None:
            return JSONResponse({"success": True, "face_found": False, "message":"Failed to crop face"}, status_code=200)

        # run FER+
        outputs = ort_sess.run(None, {"Input3": inp})  # common input name for FER+ models
        logits = outputs[0].reshape(-1)
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        label = EMOTIONS[idx]
        emoji = EMOJI_MAP[label]

        # return also bbox so frontend can draw if it wants
        return {
            "success": True,
            "face_found": True,
            "label": label,
            "emoji": emoji,
            "probabilities": {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))},
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        }

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
