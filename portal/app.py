"""
Flask web portal for the NLC Classification Pipeline.

Provides a browser-based dashboard for:
- Dataset overview and statistics
- Single-image prediction (upload or local path)
"""

import os
from pathlib import Path

from flask import Flask, render_template, request, jsonify


def create_app(project_root: str | None = None) -> Flask:
    """Create and configure the Flask application."""
    if project_root is None:
        project_root = str(Path(__file__).resolve().parent.parent)

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["PROJECT_ROOT"] = project_root

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #

    @app.route("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------------ #
    # API — dataset stats
    # ------------------------------------------------------------------ #

    @app.route("/api/dataset-stats")
    def dataset_stats():
        """Return a summary of the available data sources."""
        root = app.config["PROJECT_ROOT"]

        # Count CAS images
        cas_dir = os.path.join(root, "Cloud Appreciation Society data")
        cas_nlc, cas_neg = 0, 0
        cas_types: dict[str, int] = {}
        if os.path.isdir(cas_dir):
            for d in sorted(os.listdir(cas_dir)):
                dp = os.path.join(cas_dir, d)
                if not os.path.isdir(dp):
                    continue
                n = 0
                for dirpath, _, files in os.walk(dp):
                    n += sum(1 for f in files
                             if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"})
                cas_types[d] = n
                if d == "noctilucent":
                    cas_nlc += n
                else:
                    cas_neg += n

        # Count gallery images
        gallery_csv = os.path.join(root, "spaceweather_gallery_data.csv")
        gallery_count = 0
        if os.path.isfile(gallery_csv):
            with open(gallery_csv) as f:
                gallery_count = sum(1 for _ in f) - 1

        # Count SCW CSV rows
        scw_csv_name = ""
        scw_total, scw_nlc, scw_no = 0, 0, 0
        for fn in os.listdir(root):
            if fn.startswith("space-cloud-watch") and fn.endswith(".csv"):
                scw_csv_name = fn
                break
        if scw_csv_name:
            import pandas as pd
            scw_df = pd.read_csv(os.path.join(root, scw_csv_name))
            scw_total = len(scw_df)
            col = "did you see nlc?"
            if col in scw_df.columns:
                scw_nlc = int((scw_df[col] == "Yes").sum())
                scw_no = int((scw_df[col] == "No").sum())

        # Check model status
        checkpoint = os.path.join(root, "checkpoints", "best_model.pt")
        has_model = os.path.isfile(checkpoint)

        # Read latest training report
        training_metrics = {}
        logs_dir = os.path.join(root, "logs")
        if os.path.isdir(logs_dir):
            import json
            reports = sorted([f for f in os.listdir(logs_dir) if f.startswith("training_report")])
            if reports:
                with open(os.path.join(logs_dir, reports[-1])) as f:
                    report = json.load(f)
                training_metrics = {
                    "best_accuracy": report.get("best_val_acc"),
                    "best_f1": report.get("best_val_f1"),
                    "best_epoch": report.get("best_epoch"),
                    "total_epochs": report.get("total_epochs"),
                }

        return jsonify({
            "cas": {"nlc": cas_nlc, "non_nlc": cas_neg, "types": cas_types},
            "gallery": {"count": gallery_count},
            "scw": {"total": scw_total, "nlc": scw_nlc, "no_nlc": scw_no, "csv": scw_csv_name},
            "total_positives": cas_nlc + gallery_count + scw_nlc,
            "total_negatives": cas_neg + scw_no,
            "has_model": has_model,
            "training_metrics": training_metrics,
        })

    # ------------------------------------------------------------------ #
    # API — prediction
    # ------------------------------------------------------------------ #

    @app.route("/api/predict", methods=["POST"])
    def predict():
        """Run prediction on a single image (local path or uploaded file)."""
        root = app.config["PROJECT_ROOT"]
        checkpoint = os.path.join(root, "checkpoints", "best_model.pt")

        if not os.path.isfile(checkpoint):
            return jsonify({"error": "No trained model found. Run training first via CLI."}), 400

        image_path = None
        if request.is_json:
            image_path = request.json.get("image_path")
        elif "image" in request.files:
            f = request.files["image"]
            upload_dir = os.path.join(root, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            image_path = os.path.join(upload_dir, f.filename)
            f.save(image_path)
        elif request.form.get("image_path"):
            image_path = request.form["image_path"]

        if not image_path:
            return jsonify({"error": "No image provided"}), 400

        try:
            from nlc_classifier.config import get_default_config
            from nlc_classifier.inference import load_inference_engine

            config = get_default_config()
            engine = load_inference_engine(checkpoint, config, config.get_device())
            result = engine.predict(image_path)

            return jsonify({
                "predicted_class": result.predicted_class,
                "predicted_label": result.predicted_label,
                "confidence": round(result.confidence, 4),
                "entropy": round(result.entropy, 4),
                "probabilities": result.probabilities,
                "needs_review": result.needs_review,
                "review_reason": getattr(result, "review_reason", ""),
                "nlc_types": getattr(result, "nlc_types", []),
                "nlc_type_probabilities": {
                    k: round(v, 4) for k, v in getattr(result, "nlc_type_probabilities", {}).items()
                },
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app
