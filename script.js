const MODEL_URL = "model/face_model.json";
const LABELS_URL = "model/labels.json";

const imageInput = document.querySelector("#image-input");
const dropZone = document.querySelector("#drop-zone");
const imageCanvas = document.querySelector("#image-canvas");
const heatmapCanvas = document.querySelector("#heatmap-canvas");
const emptyState = document.querySelector("#empty-state");
const modelStatus = document.querySelector("#model-status");
const modelDetail = document.querySelector("#model-detail");
const summaryCards = document.querySelector("#summary-cards");
const qualityCards = document.querySelector("#quality-cards");
const xaiCards = document.querySelector("#xai-cards");
const questionsList = document.querySelector("#questions");
const ageGroup = document.querySelector("#age-group");
const concern = document.querySelector("#concern");
const duration = document.querySelector("#duration");
const symptoms = document.querySelector("#symptoms");

let modelReady = false;
let model = null;
let labels = [];
let lastQuality = null;

checkModelReadiness();

imageInput.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (file) {
    loadImageFile(file);
  }
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  const file = event.dataTransfer.files?.[0];
  if (file) {
    loadImageFile(file);
  }
});

[ageGroup, concern, duration, symptoms].forEach((input) => {
  input.addEventListener("input", () => {
    if (lastQuality) {
      renderQuestions(lastQuality);
    }
  });
});

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`#${button.dataset.tab}-panel`).classList.add("active");
  });
});

async function checkModelReadiness() {
  try {
    const [modelResponse, labelsResponse] = await Promise.all([fetch(MODEL_URL, { cache: "no-store" }), fetch(LABELS_URL, { cache: "no-store" })]);
    if (!modelResponse.ok || !labelsResponse.ok) {
      throw new Error("Model files missing");
    }
    model = await modelResponse.json();
    labels = await labelsResponse.json();
    modelReady = true;
    modelStatus.textContent = "Model files detected";
    modelDetail.textContent = `${labels.length} SCIN face/head labels loaded. Research prototype only.`;
  } catch (_error) {
    modelReady = false;
    modelStatus.textContent = "No trained model installed";
    modelDetail.textContent = "Add real exported files to model/face_model.json and model/labels.json to enable predictions.";
  }
}

function loadImageFile(file) {
  if (!file.type.startsWith("image/")) {
    renderSummary([
      card("danger", "Invalid file", "Please upload a JPG, PNG, or WEBP image file.", "File check"),
    ]);
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => analyzeImage(image, file);
    image.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function analyzeImage(image, file) {
  drawImage(image);
  const metrics = computeQualityMetrics();
  lastQuality = metrics;
  const prediction = modelReady ? classifyCanvas() : null;
  renderQuality(metrics, file);
  renderSummary(buildSummaryCards(metrics, prediction));
  renderXai(prediction);
  renderQuestions(metrics);
}

function drawImage(image) {
  emptyState.style.display = "none";
  const context = imageCanvas.getContext("2d", { willReadFrequently: true });
  const heatContext = heatmapCanvas.getContext("2d");
  const maxWidth = imageCanvas.width;
  const maxHeight = imageCanvas.height;
  const scale = Math.min(maxWidth / image.width, maxHeight / image.height);
  const width = Math.round(image.width * scale);
  const height = Math.round(image.height * scale);
  const x = Math.round((maxWidth - width) / 2);
  const y = Math.round((maxHeight - height) / 2);

  context.clearRect(0, 0, maxWidth, maxHeight);
  heatContext.clearRect(0, 0, maxWidth, maxHeight);
  context.fillStyle = "#121c20";
  context.fillRect(0, 0, maxWidth, maxHeight);
  context.drawImage(image, x, y, width, height);
}

function computeQualityMetrics() {
  const context = imageCanvas.getContext("2d", { willReadFrequently: true });
  const { width, height } = imageCanvas;
  const data = context.getImageData(0, 0, width, height).data;
  let total = 0;
  let totalSq = 0;
  let skinLike = 0;
  let nonBackground = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = (r + g + b) / 3;
    total += brightness;
    totalSq += brightness * brightness;

    if (brightness > 24) {
      nonBackground += 1;
    }
    if (r > 65 && g > 35 && b > 20 && r > b && Math.max(r, g, b) - Math.min(r, g, b) > 12) {
      skinLike += 1;
    }
  }

  const pixels = data.length / 4;
  const mean = total / pixels;
  const variance = totalSq / pixels - mean * mean;
  const contrast = Math.sqrt(Math.max(variance, 0));
  const skinRatio = nonBackground ? skinLike / nonBackground : 0;
  const sharpness = estimateSharpness(context, width, height);

  return {
    brightness: Math.round(mean),
    contrast: Math.round(contrast),
    sharpness: Math.round(sharpness),
    skinRatio,
    status: qualityStatus(mean, contrast, sharpness, skinRatio),
  };
}

function estimateSharpness(context, width, height) {
  const sampleWidth = 160;
  const sampleHeight = 120;
  const temp = document.createElement("canvas");
  temp.width = sampleWidth;
  temp.height = sampleHeight;
  const tempContext = temp.getContext("2d", { willReadFrequently: true });
  tempContext.drawImage(imageCanvas, 0, 0, sampleWidth, sampleHeight);
  const data = tempContext.getImageData(0, 0, sampleWidth, sampleHeight).data;
  let sum = 0;
  let count = 0;

  for (let y = 1; y < sampleHeight - 1; y += 1) {
    for (let x = 1; x < sampleWidth - 1; x += 1) {
      const center = grayAt(data, sampleWidth, x, y);
      const lap =
        4 * center -
        grayAt(data, sampleWidth, x - 1, y) -
        grayAt(data, sampleWidth, x + 1, y) -
        grayAt(data, sampleWidth, x, y - 1) -
        grayAt(data, sampleWidth, x, y + 1);
      sum += Math.abs(lap);
      count += 1;
    }
  }
  return sum / count;
}

function grayAt(data, width, x, y) {
  const index = (y * width + x) * 4;
  return (data[index] + data[index + 1] + data[index + 2]) / 3;
}

function qualityStatus(brightness, contrast, sharpness, skinRatio) {
  const issues = [];
  if (brightness < 45) issues.push("too dark");
  if (brightness > 215) issues.push("too bright");
  if (contrast < 24) issues.push("low contrast");
  if (sharpness < 6) issues.push("possibly blurry");
  if (skinRatio < 0.18) issues.push("face/skin area not clear");
  return issues;
}

function buildSummaryCards(metrics, prediction) {
  const cards = [];
  if (metrics.status.length) {
    cards.push(card("warn", "Image needs review", `Quality concerns: ${metrics.status.join(", ")}. Retake the photo before trusting any model output.`, "Quality"));
  } else {
    cards.push(card("good", "Image quality acceptable", "Lighting, contrast, sharpness, and visible skin area look acceptable for pre-screening.", "Quality"));
  }

  if (modelReady) {
    if (prediction) {
      const top = prediction[0];
      cards.push(
        card(
          "warn",
          `Detector signal: ${top.label}`,
          `${Math.round(top.score * 100)}% relative match from a tiny real SCIN face/head centroid model. This is not diagnosis.`,
          "Real-data model"
        )
      );
    } else {
      cards.push(card("warn", "Model files detected", "The model exists, but prediction could not be computed for this image.", "Model"));
    }
  } else {
    cards.push(card("neutral", "Prediction locked", "No trained model artifact is installed, so the app will not invent disease categories.", "Model"));
  }

  cards.push(card("neutral", "Face-only scope", "This project is designed for facial skin images only. Do not use it for feet, hands, or full-body lesions.", "Scope"));
  return cards;
}

function renderQuality(metrics, file) {
  const rows = [
    card(metricLevel(metrics.brightness, 45, 215), "Brightness", `${metrics.brightness}/255. Target: balanced, not shadowed or washed out.`, "Image"),
    card(metrics.contrast >= 24 ? "good" : "warn", "Contrast", `${metrics.contrast}. Low contrast can hide redness, scale, and lesion borders.`, "Image"),
    card(metrics.sharpness >= 6 ? "good" : "warn", "Sharpness", `${metrics.sharpness}. Blur weakens any model or clinician review.`, "Image"),
    card(metrics.skinRatio >= 0.18 ? "good" : "warn", "Visible facial skin", `${Math.round(metrics.skinRatio * 100)}% skin-like region estimate.`, "Face check"),
    card("neutral", "File", `${file.name} (${Math.round(file.size / 1024)} KB).`, "Upload"),
  ];
  qualityCards.innerHTML = rows.map(renderCard).join("");
}

function renderSummary(cards) {
  summaryCards.innerHTML = cards.map(renderCard).join("");
}

function renderXai(prediction) {
  const heatContext = heatmapCanvas.getContext("2d");
  heatContext.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

  if (!modelReady) {
    xaiCards.innerHTML = renderCard(
      card("neutral", "No XAI overlay", "A heatmap would be misleading without a real trained model. Export a model first, then enable saliency.", "XAI")
    );
    return;
  }

  drawPatchHeatmap();
  xaiCards.innerHTML = renderCard(
    card(
      "warn",
      "Patch heatmap generated",
      `Overlay highlights facial regions with stronger redness/texture signals for the top model class${prediction?.[0] ? ` (${prediction[0].label})` : ""}. This is explanatory support, not proof of disease.`,
      "XAI"
    )
  );
}

function classifyCanvas() {
  if (!model?.centroids || !model?.featureMean || !model?.featureStd) {
    return null;
  }
  const features = extractFeatureVectorFromCanvas(imageCanvas);
  const normalized = features.map((value, index) => (value - model.featureMean[index]) / model.featureStd[index]);
  const distances = model.labels.map((label) => ({
    label,
    distance: euclidean(normalized, model.centroids[label]),
    count: model.classCounts?.[label] || 0,
  }));
  const maxDistance = Math.max(...distances.map((item) => item.distance), 1);
  return distances
    .map((item) => ({ ...item, score: Math.max(0, 1 - item.distance / maxDistance) }))
    .sort((a, b) => b.score - a.score);
}

function extractFeatureVectorFromCanvas(canvas) {
  const sample = document.createElement("canvas");
  sample.width = 160;
  sample.height = 160;
  const sampleContext = sample.getContext("2d", { willReadFrequently: true });
  sampleContext.drawImage(canvas, 0, 0, 160, 160);
  const data = sampleContext.getImageData(0, 0, 160, 160).data;
  const channels = makeChannels(data, 160, 160);
  const features = [];
  [channels.r, channels.g, channels.b, channels.gray, channels.redness, channels.texture].forEach((channel) => {
    features.push(mean(channel), std(channel), percentile(channel, 25), percentile(channel, 75));
  });
  [channels.redness, channels.texture, channels.gray].forEach((channel) => {
    for (let y = 0; y < 4; y += 1) {
      for (let x = 0; x < 4; x += 1) {
        features.push(gridMean(channel, 160, x * 40, y * 40, 40, 40));
      }
    }
  });
  return features;
}

function makeChannels(data, width, height) {
  const r = [];
  const g = [];
  const b = [];
  const gray = [];
  const redness = [];
  for (let i = 0; i < data.length; i += 4) {
    const rv = data[i] / 255;
    const gv = data[i + 1] / 255;
    const bv = data[i + 2] / 255;
    r.push(rv);
    g.push(gv);
    b.push(bv);
    gray.push((rv + gv + bv) / 3);
    redness.push(Math.max(0, rv - (gv + bv) / 2));
  }
  const texture = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      const left = gray[y * width + Math.max(0, x - 1)];
      const right = gray[y * width + Math.min(width - 1, x + 1)];
      const up = gray[Math.max(0, y - 1) * width + x];
      const down = gray[Math.min(height - 1, y + 1) * width + x];
      texture.push(Math.min(1, Math.abs(right - left) + Math.abs(down - up)));
    }
  }
  return { r, g, b, gray, redness, texture };
}

function drawPatchHeatmap() {
  const heatContext = heatmapCanvas.getContext("2d");
  const sample = document.createElement("canvas");
  sample.width = 160;
  sample.height = 160;
  const sampleContext = sample.getContext("2d", { willReadFrequently: true });
  sampleContext.drawImage(imageCanvas, 0, 0, 160, 160);
  const data = sampleContext.getImageData(0, 0, 160, 160).data;
  const channels = makeChannels(data, 160, 160);
  heatContext.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

  for (let y = 0; y < 8; y += 1) {
    for (let x = 0; x < 8; x += 1) {
      const rednessScore = gridMean(channels.redness, 160, x * 20, y * 20, 20, 20);
      const textureScore = gridMean(channels.texture, 160, x * 20, y * 20, 20, 20);
      const score = Math.min(1, rednessScore * 2.2 + textureScore * 1.2);
      if (score < 0.12) continue;
      heatContext.fillStyle = `rgba(255, 64, 32, ${0.12 + score * 0.5})`;
      heatContext.fillRect(x * 90, y * 70, 90, 70);
    }
  }
}

function euclidean(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function std(values) {
  const avg = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - avg) ** 2, 0) / values.length);
}

function percentile(values, pct) {
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(sorted.length - 1, Math.max(0, Math.round((pct / 100) * (sorted.length - 1))));
  return sorted[index];
}

function gridMean(channel, width, startX, startY, patchWidth, patchHeight) {
  let total = 0;
  let count = 0;
  for (let y = startY; y < startY + patchHeight; y += 1) {
    for (let x = startX; x < startX + patchWidth; x += 1) {
      total += channel[y * width + x];
      count += 1;
    }
  }
  return total / count;
}

function renderQuestions(metrics) {
  const items = [];
  const concernValue = concern.value;
  const symptomText = symptoms.value.trim();

  if (metrics.status.length) {
    items.push(`Should this image be retaken because it is ${metrics.status.join(", ")}?`);
  }
  if (concernValue) {
    items.push(`For the selected concern (${concern.options[concern.selectedIndex].text}), what diagnoses should a dermatologist rule out?`);
  }
  if (duration.value) {
    items.push(`Does the duration (${duration.options[duration.selectedIndex].text}) change urgency or likely causes?`);
  }
  if (symptomText) {
    items.push(`Do these symptoms suggest urgent care or infection: ${symptomText}?`);
  }
  items.push("Are there warning signs such as eye involvement, fever, rapid spread, bleeding, severe pain, or a changing dark lesion?");
  items.push("What treatments or skin products have already been used, and could any be irritating the face?");
  items.push("Does skin tone, lighting, or camera quality limit confidence in visual assessment?");
  questionsList.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function metricLevel(value, low, high) {
  if (value < low || value > high) return "warn";
  return "good";
}

function card(level, title, message, source) {
  return { level, title, message, source };
}

function renderCard(item) {
  return `
    <article class="card ${item.level}">
      <span>${escapeHtml(item.source)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <p>${escapeHtml(item.message)}</p>
    </article>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
