import base64
import binascii
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="RSI Attendance Face Service", version="0.1.0")
_face_app = None


class RegisterRequest(BaseModel):
    images: List[str] = Field(min_length=1, max_length=5)


class VerifyRequest(BaseModel):
    image: str
    embedding: List[float] = Field(min_length=16, max_length=4096)
    model_name: Optional[str] = None


def check_service_key(service_key: Optional[str]) -> None:
    expected = os.getenv("FACE_SERVICE_SHARED_SECRET", "").strip()
    if expected and service_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def decode_image(value: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=422, detail="Gambar tidak valid")
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="Format gambar tidak valid")
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=413, detail="Ukuran gambar terlalu besar")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Gambar tidak dapat dibaca")
    return image


def get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="InsightFace belum terpasang pada face service") from exc

    providers = [item.strip() for item in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",") if item.strip()]
    model = FaceAnalysis(name=os.getenv("INSIGHTFACE_MODEL", "buffalo_l"), providers=providers)
    model.prepare(ctx_id=0, det_size=(640, 640))
    _face_app = model
    return _face_app


def one_face_embedding(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 35 or brightness > 245:
        raise HTTPException(status_code=422, detail="Pencahayaan wajah tidak sesuai")

    faces = get_face_app().get(image)
    if len(faces) == 0:
        raise HTTPException(status_code=422, detail="Wajah tidak terdeteksi")
    if len(faces) > 1:
        raise HTTPException(status_code=422, detail="Lebih dari satu wajah terdeteksi")

    face = faces[0]
    x1, y1, x2, y2 = [int(item) for item in face.bbox]
    if min(x2 - x1, y2 - y1) < int(os.getenv("FACE_MIN_SIZE", "100")):
        raise HTTPException(status_code=422, detail="Wajah terlalu jauh")
    embedding = np.asarray(face.embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise HTTPException(status_code=422, detail="Embedding wajah tidak valid")
    return embedding / norm


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.dot(left, right))


@app.get("/health")
def health():
    return {"success": True, "data": {"service": "face", "model_loaded": _face_app is not None}}


@app.post("/face/register")
def register(request: RegisterRequest, x_face_service_key: Optional[str] = Header(default=None)):
    check_service_key(x_face_service_key)
    embeddings = [one_face_embedding(decode_image(image)) for image in request.images]
    average = np.mean(np.stack(embeddings), axis=0)
    average = average / np.linalg.norm(average)
    return {
        "success": True,
        "data": {
            "embedding": average.astype(float).tolist(),
            "model_name": os.getenv("INSIGHTFACE_MODEL", "buffalo_l"),
            "samples": len(embeddings),
        },
    }


@app.post("/face/verify")
def verify(request: VerifyRequest, x_face_service_key: Optional[str] = Header(default=None)):
    check_service_key(x_face_service_key)
    current = one_face_embedding(decode_image(request.image))
    reference = np.asarray(request.embedding, dtype=np.float32)
    if reference.ndim != 1 or reference.size != current.size:
        raise HTTPException(status_code=422, detail="Model embedding tidak sesuai")
    score = similarity(current, reference)
    threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
    return {
        "success": True,
        "data": {
            "match": score >= threshold,
            "score": round(score, 5),
            "threshold": threshold,
            "model_name": os.getenv("INSIGHTFACE_MODEL", "buffalo_l"),
        },
    }
