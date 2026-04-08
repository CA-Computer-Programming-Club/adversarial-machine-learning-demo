from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
CORPUS_DIR = BASE_DIR / 'public' / 'corpus'
DIST_DIR = BASE_DIR / 'dist'

app = FastAPI(title='Animal Attack Demo API')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
loss_object = tf.keras.losses.CategoricalCrossentropy()
anomaly_model = tf.keras.Model(inputs=model.input, outputs=[model.layers[-5].output, model.layers[-3].output])


class AnalyzeRequest(BaseModel):
    corpusImage: str | None = None
    image: str | None = None
    attack: str = 'fgsm'
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
    return [{"id": item[0], "label": item[1], "confidence": float(item[2])} for item in decoded]


def create_adversarial_pattern(input_image: tf.Tensor, input_label: tf.Tensor) -> tf.Tensor:
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


def iterative_attack(image: tf.Tensor, target_index: int, epsilon: float, alpha: float, steps: int) -> tf.Tensor:
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


def calculate_anomaly_score(image: tf.Tensor) -> float:
    feature_maps = anomaly_model(image)
    scores = []
    for fmap in feature_maps:
        mean_score = tf.reduce_mean(fmap)
        var_score = tf.math.reduce_variance(fmap)
        scores.append(mean_score + var_score)
    return float(tf.reduce_mean(scores).numpy())


def read_image_bytes(data: str | None, corpus_name: str | None) -> np.ndarray:
    try:
        if data:
            if ',' in data:
                data = data.split(',', 1)[1]
            raw = base64.b64decode(data)
            img = Image.open(io.BytesIO(raw)).convert('RGB')
        elif corpus_name:
            img = Image.open(CORPUS_DIR / corpus_name).convert('RGB')
        else:
            raise ValueError('No image provided')
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'Could not read image: {exc}') from exc
    return np.array(img).astype('float32')


def to_display_png(image: tf.Tensor) -> str:
    arr = image[0].numpy()
    arr = ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype('uint8')
    img = Image.fromarray(arr).resize((448, 448))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


@app.get('/api/corpus')
def corpus():
    files = []
    for path in sorted(CORPUS_DIR.glob('*')):
        if path.is_file():
            files.append({'name': path.name, 'url': f'/corpus/{path.name}'})
    return {'images': files}


@app.post('/api/analyze')
def analyze(payload: AnalyzeRequest):
    image_arr = read_image_bytes(payload.image, payload.corpusImage)
    image = preprocess(image_arr)
    base_probs = model.predict(image, verbose=0)
    base_top = decode_top(base_probs, top=5)
    target_index = int(np.argmax(base_probs[0]))

    if payload.attack == 'fgsm':
        adv = fgsm_attack(image, target_index, payload.epsilon)
    else:
        adv = iterative_attack(image, target_index, payload.epsilon, payload.alpha, payload.steps)

    adv_probs = model.predict(adv, verbose=0)
    adv_top = decode_top(adv_probs, top=5)

    original_anomaly = calculate_anomaly_score(image)
    attacked_anomaly = calculate_anomaly_score(adv)

    return {
        'baseImage': to_display_png(image),
        'attackedImage': to_display_png(adv),
        'baseTop': base_top,
        'attackedTop': adv_top,
        'epsilon': payload.epsilon,
        'steps': payload.steps,
        'alpha': payload.alpha,
        'attack': payload.attack,
        'anomaly': {
            'original': original_anomaly,
            'attacked': attacked_anomaly,
            'delta': attacked_anomaly - original_anomaly,
            'thresholdFlagged': attacked_anomaly > original_anomaly + 0.05,
        },
    }


@app.get('/corpus/{name}')
def corpus_asset(name: str):
    path = CORPUS_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail='Image not found')
    return FileResponse(path)


@app.get('/')
def root():
    return FileResponse(DIST_DIR / 'index.html')


app.mount('/', StaticFiles(directory=DIST_DIR, html=True), name='static')
