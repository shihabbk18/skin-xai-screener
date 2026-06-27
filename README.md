# FaceSkin Lens

FaceSkin Lens is a face-only facial skin pre-screening scaffold for real-data dermatology AI and explainability.

It does **not** ship fake predictions. The browser app checks image quality immediately, but disease categories and heatmaps stay locked until a real trained model is exported into `model/model.json` and `model/labels.json`.

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

## Training pipeline

The `scripts/prepare_face_dataset.py` and `scripts/train_face_model.py` files are starter scripts for a real training path. They expect real downloaded dataset files; they do not generate fake data.

## Safety

This app must not be used as a medical diagnosis tool. Urgent symptoms such as eye involvement, severe pain, fever with rash, facial swelling with breathing trouble, rapidly spreading infection, uncontrolled bleeding, or a rapidly changing dark lesion require medical care.
