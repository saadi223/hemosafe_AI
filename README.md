# AI-Based Blood Damage Prediction Model for Internal Artificial Organs

**HemoPredict AI** | Muhammad Saad Hanif | Alibaba Cloud AI Hackathon 2026 (research prototype)

An interactive Streamlit app running a **saved, trained Extra Trees regression pipeline** to predict `target_log10_hi`, a **CFD-derived log10 hemolysis-index indicator**. This is not a clinical hemolysis predictor.

## Live demo workflow

1. Choose **Verified dataset example**.
2. Select one of four supported geometry families: `baseline`, `herringbone`, `reduced_gap`, `reducedport`.
3. Select an available flow rate and click **Run real model prediction**.
4. Inspect predicted `log10(HI)`, stored dataset target, and absolute error.
5. Scroll down for diagnostics, feature importance, and leave-one-device-out results.

**Important:** Dataset examples may include training samples. Their displayed errors are **not** independent test scores. For manual prediction, all CFD shear, exposure, and physics-index inputs are required; geometry and flow alone are insufficient.

## Results and scope

| Evaluation | R² | MAE | RMSE |
|---|---:|---:|---:|
| Development holdout (94 samples, new flows in seen geometry families) | 0.9818 | 0.1201 | 0.3172 |

These metrics are in **log10(HI)** target units (MAE/RMSE), not percentage accuracy or clinical validation. Generalization to an entirely unseen device family is substantially weaker (see `unseen_device_metrics.csv`). No fifth device geometry is supported. The data are physics-informed and inspired by a *Micromachines* paper; they are **not a verified exact reproduction of its supplementary measurements**.

## Run locally

Requires Python 3.10+ (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload **all files and the `.streamlit` folder** in this directory to the repository root.
2. At https://share.streamlit.io/ choose **Create app**, then select the repository, branch, and `app.py`.
3. Deploy. The model is included; no API key or Alibaba Cloud account is required.
4. Open the resulting `.streamlit.app` URL and test a prediction.

**Hosting disclosure:** This workflow deploys to **Streamlit Community Cloud, not Alibaba Cloud**. Do not imply otherwise.

## Responsible use

- Research demonstration only; not for medical decision-making.
- The model uses a physics-informed dataset and has no independent external CFD/experimental or clinical validation.
- The `model.joblib` file is part of this trusted project package; **never load joblib models from untrusted sources**.
- See the original FYP documentation and source study for context and scientific limitations.
