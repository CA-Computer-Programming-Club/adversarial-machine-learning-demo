from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path
import binascii

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "public" / "corpus"
DIST_DIR = BASE_DIR / "dist"

app = FastAPI(title="Adversarial Machine Learning Demo API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "https://math-expo.cacpc.dev"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

model = tf.keras.applications.MobileNetV2(include_top=True, weights="imagenet")
model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()

_ML_SEMAPHORE = asyncio.Semaphore(2)


class AnalyzeRequest(BaseModel):
    corpusImage: str | None = None
    image: str | None = None
    attack: str = "fgsm"
    epsilon: float = 0.1
    steps: int = 5
    alpha: float = 0.02


def preprocess(image_arr: np.ndarray) -> tf.Tensor:
    image = tf.convert_to_tensor(image_arr, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image[None, ...]


def decode_top(probs: np.ndarray, top: int = 5):
    decoded = decode_predictions(probs, top=top)[0]
    return [
        {"id": item[0], "label": item[1], "confidence": float(item[2])}
        for item in decoded
    ]


def create_adversarial_pattern(
    input_image: tf.Tensor, input_label: tf.Tensor
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)
    gradient = tape.gradient(loss, input_image)
    return tf.sign(gradient)


def fgsm_attack(image: tf.Tensor, target_index: int, epsilon: float) -> tf.Tensor:
    label = tf.one_hot(target_index, 1000)
    label = tf.reshape(label, (1, 1000))
    perturbations = create_adversarial_pattern(image, label)
    adv = image + epsilon * perturbations
    return tf.clip_by_value(adv, -1, 1)


def iterative_attack(
    image: tf.Tensor, target_index: int, epsilon: float, alpha: float, steps: int
) -> tf.Tensor:
    original = tf.identity(image)
    adv = tf.identity(image)
    label = tf.one_hot(target_index, 1000)
    label = tf.reshape(label, (1, 1000))
    for _ in range(steps):
        perturbations = create_adversarial_pattern(adv, label)
        adv = adv + alpha * perturbations
        adv = tf.minimum(tf.maximum(adv, original - epsilon), original + epsilon)
        adv = tf.clip_by_value(adv, -1, 1)
    return adv


def read_image_bytes(data: str | None, corpus_name: str | None) -> np.ndarray:
    try:
        if data:
            if "," in data:
                data = data.split(",", 1)[1]
            try:
                raw = base64.b64decode(data, validate=True)
            except binascii.Error as exc:
                raise ValueError("Uploaded or pasted image data is not valid base64") from exc
            img = Image.open(io.BytesIO(raw))
        elif corpus_name:
            img = Image.open(CORPUS_DIR / corpus_name)
        else:
            raise ValueError("No image provided")

        img.load()
        if img.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            img = img.convert("RGBA")

        if img.mode in {"RGBA", "LA"}:
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            background.alpha_composite(img.convert("RGBA"))
            img = background.convert("RGB")
        elif img.mode == "P":
            img = img.convert("RGBA").convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")
        else:
            img = img.convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image. Please use a valid PNG, JPG, WEBP, GIF, or other standard image format. Technical detail: {exc}",
        ) from exc
    return np.array(img).astype("float32")


def to_display_png(image: tf.Tensor) -> str:
    arr = image[0].numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr).resize((448, 448))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def to_difference_png(original: tf.Tensor, adv: tf.Tensor, epsilon: float) -> str:
    """Absolute per-channel pixel difference, scaled relative to epsilon.

    Pixels perturbed by the full epsilon appear at ~170/255 brightness.
    Pixels that were clipped at the [-1, 1] boundary appear darker,
    revealing spatial structure where the original image was near an extreme.
    """
    orig_arr = original[0].numpy().astype("float32")
    adv_arr = adv[0].numpy().astype("float32")
    diff = np.abs(adv_arr - orig_arr)
    diff_scaled = np.clip(diff / (epsilon * 1.5 + 1e-8), 0.0, 1.0)
    img = Image.fromarray((diff_scaled * 255.0).astype("uint8")).resize((448, 448))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/corpus")
def corpus():
    files = []
    for path in sorted(CORPUS_DIR.glob("*")):
        if path.is_file():
            files.append({"name": path.name, "url": f"/corpus/{path.name}"})
    return {"images": files}


def _run_analyze(payload: AnalyzeRequest) -> dict:
    image_arr = read_image_bytes(payload.image, payload.corpusImage)
    image = preprocess(image_arr)
    base_probs = model.predict(image, verbose=0)
    base_top = decode_top(base_probs, top=5)
    target_index = int(np.argmax(base_probs[0]))

    if payload.attack == "fgsm":
        adv = fgsm_attack(image, target_index, payload.epsilon)
    else:
        adv = iterative_attack(
            image, target_index, payload.epsilon, payload.alpha, payload.steps
        )

    adv_probs = model.predict(adv, verbose=0)
    adv_top = decode_top(adv_probs, top=5)

    return {
        "baseImage": to_display_png(image),
        "perturbedImage": to_display_png(adv),
        "differenceImage": to_difference_png(image, adv, payload.epsilon),
        "baseTop": base_top,
        "perturbedTop": adv_top,
        "epsilon": payload.epsilon,
        "steps": payload.steps,
        "alpha": payload.alpha,
        "attack": payload.attack,
    }


@app.post("/api/analyze")
async def analyze(payload: AnalyzeRequest):
    async with _ML_SEMAPHORE:
        return await asyncio.to_thread(_run_analyze, payload)


@app.get("/corpus/{name}")
def corpus_asset(name: str):
    path = CORPUS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/")
def root():
    return FileResponse(DIST_DIR / "index.html")


app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="static")
