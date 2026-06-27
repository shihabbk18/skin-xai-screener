# FaceSkin Lens

FaceSkin Lens is a face-only facial skin pre-screening scaffold for real-data dermatology AI and explainability.

It does **not** ship fake predictions. The browser app checks image quality immediately. When `model/face_model.json` and `model/labels.json` are present, it loads a small real-data SCIN face/head centroid model and shows research-grade detector signals plus a patch heatmap.

## Scope

- Face images only.
- Screening support only, not diagnosis.
- Designed for real datasets such as SCIN and Fitzpatrick17k after face-only filtering.

## Run locally

```powershell
cd "C:\Users\NuRuL AzAm\Documents\SHIHAB\skin-xai-screener"
py -m http.server 5180 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:5180/
```

## Real data plan

Recommended datasets:

- SCIN: Google Research Skin Condition Image Network. Use only face/head images after metadata filtering.
- Fitzpatrick17k: clinical dermatology image dataset with Fitzpatrick skin type annotations. Use only face-relevant images/classes after filtering.
- SkinCon: concept annotations for interpretability research.

## Included real-data prototype model

The current repository includes a tiny model trained from real SCIN head/neck images:

- Urticaria
- Eczema
- Allergic Contact Dermatitis

This model is intentionally labeled as a research prototype. It is trained from a very small bounded sample so it can be built locally in this environment. It is not clinically reliable and must not be represented as a diagnostic model.

## Training pipeline

The `scripts/train_scin_centroid_model.py` script downloads public SCIN metadata and real image files from Google Cloud Storage, filters to `body_parts_head_or_neck == YES`, trains the lightweight browser model, and exports:

- `model/face_model.json`
- `model/labels.json`

The `scripts/prepare_face_dataset.py` and `scripts/train_face_model.py` files remain starter scripts for a deeper TensorFlow/TensorFlow.js model when the full dependency stack is available.

## Safety

This app must not be used as a medical diagnosis tool. Urgent symptoms such as eye involvement, severe pain, fever with rash, facial swelling with breathing trouble, rapidly spreading infection, uncontrolled bleeding, or a rapidly changing dark lesion require medical care.
