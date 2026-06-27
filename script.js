const MODEL_URL = "model/model.json";
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
    labels = await labelsResponse.json();
    modelReady = true;
    modelStatus.textContent = "Model files detected";
    modelDetail.textContent = `${labels.length} labels available. Prediction requires TensorFlow.js runtime integration.`;
  } catch (_error) {
    modelReady = false;
    modelStatus.textContent = "No trained model installed";
    modelDetail.textContent = "Add real exported files to model/model.json and model/labels.json to enable predictions.";
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
  renderQuality(metrics, file);
  renderSummary(buildSummaryCards(metrics));
  renderXai();
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

function buildSummaryCards(metrics) {
  const cards = [];
  if (metrics.status.length) {
    cards.push(card("warn", "Image needs review", `Quality concerns: ${metrics.status.join(", ")}. Retake the photo before trusting any model output.`, "Quality"));
  } else {
    cards.push(card("good", "Image quality acceptable", "Lighting, contrast, sharpness, and visible skin area look acceptable for pre-screening.", "Quality"));
  }

  if (modelReady) {
    cards.push(card("warn", "Model files detected", "A real model appears installed, but this UI still needs final runtime validation before clinical use.", "Model"));
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

function renderXai() {
  const heatContext = heatmapCanvas.getContext("2d");
  heatContext.clearRect(0, 0, heatmapCanvas.width, heatmapCanvas.height);

  if (!modelReady) {
    xaiCards.innerHTML = renderCard(
      card("neutral", "No XAI overlay", "A heatmap would be misleading without a real trained model. Export a model first, then enable saliency.", "XAI")
    );
    return;
  }

  xaiCards.innerHTML = renderCard(
    card("warn", "Model detected, XAI pending", "The next engineering step is TensorFlow.js saliency or Grad-CAM export validation against the trained model.", "XAI")
  );
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
