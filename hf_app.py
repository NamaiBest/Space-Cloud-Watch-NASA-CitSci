"""
Hugging Face Spaces entry point.

Loads the NLC model once at startup, then serves:
  GET  /           -> public prediction UI
  POST /predict    -> JSON prediction API (image upload)
"""

import os
import sys
import io
import json
import base64
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from PIL import Image

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── Load model once at startup ───────────────────────────────────────────────
print("Loading NLC model…")
try:
    from nlc_classifier.inference import load_inference_engine
    ENGINE = load_inference_engine(
        str(ROOT / "checkpoints" / "best_model.pt"),
        device="cpu"
    )
    MODEL_LOADED = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"WARNING: Model failed to load — {e}")
    ENGINE = None
    MODEL_LOADED = False

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── HTML UI (self-contained, no template files needed) ───────────────────────
UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NLC Detector — NASA Space Cloud Watch</title>
<meta name="description" content="Upload a sky photo and find out if it contains noctilucent clouds (NLC). Powered by EfficientNet-B0, trained on NASA Space Cloud Watch citizen science data.">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080c18;--surface:rgba(16,24,48,.7);--border:rgba(100,150,255,.14);
  --accent:#5ba4f5;--accent2:#a78bfa;--green:#34d399;--red:#f87171;
  --warn:#fbbf24;--text:#e2e8f0;--dim:#94a3b8;--r:16px;
  --font:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;scroll-behavior:smooth}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh;
  background-image:
    radial-gradient(ellipse 80% 50% at 10% -10%,rgba(91,164,245,.1) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 90% 110%,rgba(167,139,250,.08) 0%,transparent 60%);
}

/* ── Header ── */
header{
  text-align:center;padding:3rem 1rem 2rem;
  border-bottom:1px solid var(--border);
}
.badge{
  display:inline-flex;align-items:center;gap:.4rem;
  background:rgba(91,164,245,.1);border:1px solid rgba(91,164,245,.25);
  border-radius:999px;padding:.3rem .9rem;font-size:.75rem;font-weight:600;
  color:var(--accent);margin-bottom:1.2rem;letter-spacing:.04em;
}
header h1{font-size:clamp(1.8rem,5vw,3rem);font-weight:700;letter-spacing:-.03em;line-height:1.15;margin-bottom:.6rem;}
header h1 span{background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
header p{color:var(--dim);font-size:.95rem;max-width:520px;margin:0 auto;}

/* ── Main ── */
main{max-width:1080px;margin:0 auto;padding:2.5rem 1.2rem 5rem;}

/* ── Two-column layout ── */
.layout{display:grid;grid-template-columns:280px 1fr;gap:1.5rem;align-items:start;}
@media(max-width:720px){.layout{grid-template-columns:1fr;}}

/* ── Specs sidebar ── */
.specs-card{
  background:var(--surface);backdrop-filter:blur(16px);
  border:1px solid var(--border);border-radius:var(--r);padding:1.4rem;
  position:sticky;top:80px;
}
.specs-title{
  font-size:.7rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.08em;color:var(--accent);margin-bottom:1rem;
}
.spec-row{
  display:flex;flex-direction:column;gap:.1rem;
  padding:.55rem 0;border-bottom:1px solid rgba(255,255,255,.05);
}
.spec-row:last-child{border-bottom:none;}
.spec-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);}
.spec-value{font-size:.88rem;font-weight:500;color:var(--text);}
.spec-value.highlight{color:var(--green);}
.spec-divider{
  margin:1rem 0 .75rem;
  font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;
  color:var(--accent2);font-weight:700;
}

/* ── Upload card ── */
.card{
  background:var(--surface);backdrop-filter:blur(16px);
  border:1px solid var(--border);border-radius:var(--r);padding:2rem;
  margin-bottom:1.5rem;
}
.card-title{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-bottom:1.2rem;}

.drop-zone{
  border:2px dashed var(--border);border-radius:12px;
  padding:3rem 2rem;text-align:center;cursor:pointer;
  transition:border-color .25s,background .25s;
  position:relative;
}
.drop-zone:hover,.drop-zone.drag-over{
  border-color:var(--accent);background:rgba(91,164,245,.05);
}
.drop-zone input[type=file]{
  position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;
}
.drop-icon{font-size:2.5rem;margin-bottom:.75rem;display:block;}
.drop-label{font-size:.95rem;color:var(--dim);}
.drop-label strong{color:var(--text);}
.drop-sub{font-size:.78rem;color:var(--dim);margin-top:.35rem;}

/* Preview */
#preview-wrap{display:none;margin-top:1.2rem;text-align:center;}
#preview-img{max-width:100%;max-height:320px;border-radius:10px;border:1px solid var(--border);object-fit:contain;}

/* Button */
.btn{
  display:inline-flex;align-items:center;gap:.5rem;
  padding:.7rem 2rem;border:none;border-radius:10px;
  font-family:var(--font);font-size:.95rem;font-weight:600;color:#fff;
  background:linear-gradient(135deg,var(--accent),var(--accent2));
  cursor:pointer;transition:opacity .2s,transform .2s;width:100%;
  justify-content:center;margin-top:1rem;
}
.btn:hover{opacity:.88;transform:translateY(-1px);}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none;}

/* Spinner */
.spinner{
  width:18px;height:18px;border:2px solid rgba(255,255,255,.3);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none;
}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Result card ── */
#result-card{display:none;}

.verdict{
  display:flex;align-items:center;gap:1rem;
  padding:1.2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;
}
.verdict.nlc{background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);}
.verdict.no-nlc{background:rgba(248,113,113,.1);border:1px solid rgba(248,113,113,.25);}
.verdict-icon{font-size:2rem;line-height:1;}
.verdict-label{font-size:1.4rem;font-weight:700;}
.verdict.nlc .verdict-label{color:var(--green);}
.verdict.no-nlc .verdict-label{color:var(--red);}
.verdict-conf{font-size:.85rem;color:var(--dim);margin-top:.15rem;}

/* Metrics row */
.metrics{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.metric{
  flex:1;min-width:120px;
  background:rgba(255,255,255,.04);border:1px solid var(--border);
  border-radius:10px;padding:.85rem 1rem;
}
.metric-label{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--dim);margin-bottom:.3rem;}
.metric-value{font-size:1.15rem;font-weight:600;color:var(--text);}
.metric-value.ok{color:var(--green);}
.metric-value.warn{color:var(--warn);}

/* Type bars */
.section-label{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--dim);margin-bottom:.85rem;}
.type-bar{display:flex;align-items:center;gap:.65rem;margin-bottom:.55rem;}
.type-name{min-width:130px;font-size:.82rem;color:var(--dim);}
.type-name.detected{color:var(--green);font-weight:600;}
.bar-track{flex:1;height:7px;background:rgba(255,255,255,.07);border-radius:4px;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;transition:width .5s ease;}
.bar-fill.detected{background:var(--green);}
.bar-fill.not-detected{background:var(--accent);}
.type-pct{min-width:44px;text-align:right;font-size:.8rem;font-weight:600;}

/* Detected tag */
.detected-tags{margin-top:.75rem;display:flex;flex-wrap:wrap;gap:.4rem;}
.tag{
  padding:.25rem .7rem;border-radius:6px;font-size:.75rem;font-weight:600;
  background:rgba(52,211,153,.12);color:var(--green);border:1px solid rgba(52,211,153,.25);
}

/* Warning box */
.review-box{
  display:flex;align-items:flex-start;gap:.75rem;
  background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.2);
  border-radius:10px;padding:1rem 1.2rem;margin-top:1rem;font-size:.85rem;
}
.review-icon{font-size:1.1rem;flex-shrink:0;}

/* Disclaimer */
.disclaimer{
  margin-top:1.5rem;padding:1rem 1.2rem;
  background:rgba(255,255,255,.03);border-radius:10px;
  font-size:.78rem;color:var(--dim);line-height:1.6;
  border-left:3px solid var(--border);
}

/* Error */
.error-box{
  background:rgba(248,113,113,.08);border:1px solid rgba(248,113,113,.2);
  border-radius:10px;padding:1rem 1.2rem;font-size:.88rem;color:var(--red);
}

/* Footer */
footer{text-align:center;padding:2rem;font-size:.75rem;color:var(--dim);}
footer a{color:var(--accent);text-decoration:none;}
</style>
</head>
<body>

<header>
  <div class="badge">NASA Space Cloud Watch &nbsp;·&nbsp; EfficientNet-B0</div>
  <h1>Noctilucent Cloud<br><span>Detector</span></h1>
  <p>Upload a sky photo and find out whether noctilucent clouds (NLC) are present, along with NLC type classification.</p>
</header>

<main>
<div class="layout">

  <!-- LEFT: Model Specs sidebar -->
  <aside class="specs-card">
    <div class="specs-title">Model Specs</div>

    <div class="spec-row">
      <span class="spec-label">Architecture</span>
      <span class="spec-value">EfficientNet-B0</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Parameters</span>
      <span class="spec-value">5.3M</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Input Size</span>
      <span class="spec-value">224 × 224 px</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Backbone</span>
      <span class="spec-value">ImageNet pretrained</span>
    </div>

    <div class="spec-divider">Performance</div>
    <div class="spec-row">
      <span class="spec-label">Val Accuracy</span>
      <span class="spec-value highlight">91.0%</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">F1 Score</span>
      <span class="spec-value highlight">0.907</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Precision</span>
      <span class="spec-value">0.940</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Recall</span>
      <span class="spec-value">0.876</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Best Epoch</span>
      <span class="spec-value">28 / 30</span>
    </div>

    <div class="spec-divider">Training Data</div>
    <div class="spec-row">
      <span class="spec-label">Total Images</span>
      <span class="spec-value">890</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">NLC Positive</span>
      <span class="spec-value">445</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Non-NLC</span>
      <span class="spec-value">445</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Sources</span>
      <span class="spec-value">3 (SCW, Gallery, CAS)</span>
    </div>

    <div class="spec-divider">Tasks</div>
    <div class="spec-row">
      <span class="spec-label">Task 1</span>
      <span class="spec-value">NLC Detection (binary)</span>
    </div>
    <div class="spec-row">
      <span class="spec-label">Task 2</span>
      <span class="spec-value">NLC Type (4-class multi-label)</span>
    </div>
  </aside>

  <!-- RIGHT: Upload + results -->
  <div>
  <!-- Upload card -->
  <div class="card">
    <div class="card-title">Upload Image</div>
    <div class="drop-zone" id="drop-zone">
      <input type="file" id="file-input" accept="image/*">
      <span class="drop-icon">☁️</span>
      <p class="drop-label"><strong>Click to upload</strong> or drag & drop</p>
      <p class="drop-sub">JPG, PNG, WebP — max 10 MB</p>
    </div>
    <div id="preview-wrap">
      <img id="preview-img" src="" alt="Preview">
    </div>
    <button class="btn" id="classify-btn" onclick="classify()" disabled>
      <span class="spinner" id="spinner"></span>
      <span id="btn-text">Classify Image</span>
    </button>
  </div>

  <!-- Result card -->
  <div class="card" id="result-card">
    <div class="card-title">Result</div>

    <div id="result-inner"></div>
  </div>
  </div><!-- end right column -->
</div><!-- end layout -->
</main>

<footer>Built for NASA Space Cloud Watch
  &nbsp;·&nbsp; Model trained on 890 images (EfficientNet-B0, 91% accuracy)
</footer>

<script>
const fileInput = document.getElementById('file-input');
const dropZone  = document.getElementById('drop-zone');
const classifyBtn = document.getElementById('classify-btn');
const spinner   = document.getElementById('spinner');
const btnText   = document.getElementById('btn-text');
const previewWrap = document.getElementById('preview-wrap');
const previewImg  = document.getElementById('preview-img');
const resultCard  = document.getElementById('result-card');
const resultInner = document.getElementById('result-inner');

let selectedFile = null;

/* File selection */
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.style.display = 'block';
  classifyBtn.disabled = false;
  resultCard.style.display = 'none';
}

/* Classify */
async function classify() {
  if (!selectedFile) return;

  classifyBtn.disabled = true;
  spinner.style.display = 'inline-block';
  btnText.textContent = 'Analysing…';
  resultCard.style.display = 'none';

  const fd = new FormData();
  fd.append('image', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const d   = await res.json();

    if (d.error) {
      showError(d.error);
    } else {
      showResult(d);
    }
  } catch (e) {
    showError('Network error — please try again.');
  } finally {
    classifyBtn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = 'Classify Image';
  }
}

function showError(msg) {
  resultInner.innerHTML = `<div class="error-box">Error: ${msg}</div>`;
  resultCard.style.display = 'block';
  resultCard.scrollIntoView({behavior:'smooth',block:'start'});
}

function showResult(d) {
  const isNLC   = d.predicted_class === 1;
  const confPct = (d.confidence * 100).toFixed(1);
  const reviewOk = !d.needs_review;

  /* Verdict */
  let html = `
    <div class="verdict ${isNLC ? 'nlc' : 'no-nlc'}">
      <div class="verdict-icon">${isNLC ? '🌌' : '✗'}</div>
      <div>
        <div class="verdict-label">${d.predicted_label}</div>
        <div class="verdict-conf">${confPct}% confidence</div>
      </div>
    </div>`;

  /* Metrics */
  html += `<div class="metrics">
    <div class="metric">
      <div class="metric-label">Confidence</div>
      <div class="metric-value ${d.confidence >= 0.85 ? 'ok' : d.confidence < 0.65 ? 'warn' : ''}">${confPct}%</div>
    </div>
    <div class="metric">
      <div class="metric-label">Entropy</div>
      <div class="metric-value">${d.entropy.toFixed(3)}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Review</div>
      <div class="metric-value ${reviewOk ? 'ok' : 'warn'}">${reviewOk ? 'Auto-accepted' : 'Recommended'}</div>
    </div>
  </div>`;

  /* NLC type probabilities — only when NLC detected */
  if (isNLC && d.nlc_type_probabilities && Object.keys(d.nlc_type_probabilities).length) {
    html += `<div class="section-label">NLC Type Probabilities</div>`;
    for (const [name, prob] of Object.entries(d.nlc_type_probabilities)) {
      const pct = (prob * 100).toFixed(1);
      const det = prob > 0.5;
      html += `
        <div class="type-bar">
          <span class="type-name ${det ? 'detected' : ''}">${det ? '✓ ' : ''}${name}</span>
          <div class="bar-track"><div class="bar-fill ${det ? 'detected' : 'not-detected'}" style="width:${pct}%"></div></div>
          <span class="type-pct">${pct}%</span>
        </div>`;
    }
    if (d.nlc_types && d.nlc_types.length) {
      html += `<div class="detected-tags">`;
      for (const t of d.nlc_types) html += `<span class="tag">${t}</span>`;
      html += `</div>`;
    }
  }

  /* Human review notice */
  if (d.needs_review) {
    html += `
      <div class="review-box">
        <span class="review-icon">⚠</span>
        <span>Human review recommended — ${d.review_reason.replace(/_/g,' ')} (confidence below threshold).</span>
      </div>`;
  }

  /* Disclaimer */
  html += `
    <div class="disclaimer">
      This is an assistive AI prediction, not a definitive scientific determination.
      For research or scientific use, please have predictions verified by a trained observer.
      Model accuracy: 91% on held-out validation set (890 training images).
    </div>`;

  resultInner.innerHTML = html;
  resultCard.style.display = 'block';
  resultCard.scrollIntoView({behavior:'smooth',block:'start'});
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(UI)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODEL_LOADED:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 503

    # Accept uploaded file
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    try:
        result = ENGINE.predict(img)
        return jsonify({
            "predicted_class":        result.predicted_class,
            "predicted_label":        result.predicted_label,
            "confidence":             round(result.confidence, 4),
            "entropy":                round(result.entropy, 4),
            "probabilities":          result.probabilities,
            "needs_review":           result.needs_review,
            "review_reason":          getattr(result, "review_reason", ""),
            "nlc_types":              getattr(result, "nlc_types", []),
            "nlc_type_probabilities": {
                k: round(v, 4)
                for k, v in getattr(result, "nlc_type_probabilities", {}).items()
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    app.run(host=host, port=port)
