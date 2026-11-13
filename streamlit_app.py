"""
Pediatric pneumonia X-ray demo (educational only).
Run with:  streamlit run app.py

Frozen settings come from the team's final evaluation (MLSN Team 7).
Items marked TODO need confirming with Nico.
"""
import os
import glob
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---------------- FROZEN SETTINGS ----------------
WEIGHTS_PATH = "weights/best_model.pt"   # TODO: final weights file (LR 3e-5, seed 42, 15 epochs)
MODEL_NAME = "DenseNet121"
IMG_SIZE = 64
LEARNING_RATE = 3e-5
SEED = 42
EPOCHS = 15
CLASS_WEIGHTS = "None (unweighted)"
TRAIN_AUGMENTATION = "TODO: e.g. random flip, small rotation"   # fill in from the training notebook
THRESHOLD = 0.5                          # TODO: confirm the frozen threshold used for the test run
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]   # TODO: must match training normalization
SAMPLES_DIR = "samples"
# -------------------------------------------------

# ---------------- RESULTS (from the single test run) ----------------
RESULTS = pd.DataFrame(
    {
        "Recall": [0.9871, 0.9949],
        "Specificity": [0.9630, 0.6624],
        "AUC": [0.9981, 0.9830],
        "Accuracy": [0.9809, 0.8702],
    },
    index=["Validation", "Test"],
)
TP, TN, FP, FN = 388, 155, 79, 2
# ---------------------------------------------------------------------

DISCLAIMER = (
    "Educational demo, not for clinical use. "
    "Trained on pediatric chest X-rays (PneumoniaMNIST) from a single source."
)


def build_model():
    """Must match the training architecture exactly."""
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)  # 1 logit
    return model


def find_weights():
    """Look for the weights file in common places, then anywhere in the repo."""
    for c in [WEIGHTS_PATH, "best_model.pt", "best_model.pth", "weights/best_model.pth"]:
        if os.path.exists(c):
            return c
    found = glob.glob("**/*.pt", recursive=True) + glob.glob("**/*.pth", recursive=True)
    return found[0] if found else None


@st.cache_resource
def load_model():
    """Returns (model, error_message). Never crashes the page."""
    path = find_weights()
    if path is None:
        return None, "No weights file (.pt or .pth) found in the GitHub repo. Upload best_model.pt."
    try:
        model = build_model()
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            for k in ("model_state_dict", "state_dict"):
                if k in state:
                    state = state[k]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        return model, None
    except Exception as e:
        return None, f"Could not load weights from '{path}': {e}"


# Same as evaluation: grayscale -> resize -> 3 channels -> normalize. No augmentation.
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def predict(model, image):
    x = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()   # P(pneumonia)
    return ("Pneumonia" if prob >= THRESHOLD else "Normal"), prob


def true_label_from_name(name):
    n = name.lower()
    if "pneu" in n:
        return "Pneumonia"
    if "normal" in n:
        return "Normal"
    return None


# ------------------------------- UI ---------------------------------
st.set_page_config(page_title="Pediatric Pneumonia Demo", page_icon="🫁", layout="centered")
st.title("Pediatric Chest X-ray: Pneumonia Detector")
st.warning(DISCLAIMER)

tab_predict, tab_results, tab_about = st.tabs(["Try the model", "Test results", "Settings & limitations"])

# ---------------- Tab 1: predict ----------------
with tab_predict:
    model, load_err = load_model()
    if load_err:
        st.error(load_err)
        st.stop()
    image, true_label = None, None

    source = st.radio("Choose an input", ["Sample image", "Upload your own"], horizontal=True)
    if source == "Upload your own":
        f = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
        if f:
            image = Image.open(f).convert("L")
    else:
        if os.path.isdir(SAMPLES_DIR):
            files = sorted(x for x in os.listdir(SAMPLES_DIR) if x.lower().endswith((".png", ".jpg", ".jpeg")))
            choice = st.selectbox("Pick a sample test image", files)
            if choice:
                image = Image.open(os.path.join(SAMPLES_DIR, choice)).convert("L")
                true_label = true_label_from_name(choice)
        else:
            st.info(f"Add test images to the '{SAMPLES_DIR}' folder.")

    if image is not None:
        label, prob = predict(model, image)
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption="Input image", use_container_width=True)
        with c2:
            st.metric("Prediction", label)
            st.metric("P(pneumonia)", f"{prob:.1%}")
            st.caption(f"Decision threshold: {THRESHOLD}")
            if true_label:
                ok = "✅ correct" if true_label == label else "❌ incorrect"
                st.write(f"True label: **{true_label}** ({ok})")

        st.info(
            "Context: on the held-out test set this model caught almost all pneumonia cases "
            "(recall 99.5%) but wrongly flagged about a third of normal X-rays "
            "(specificity 66.2%). A 'Pneumonia' result is more likely to be a false alarm than a "
            "'Normal' result is to be a miss."
        )

        # --- Optional Grad-CAM (when Yuki's function is ready) ---
        # from gradcam import make_overlay
        # if st.checkbox("Show Grad-CAM overlay"):
        #     st.image(make_overlay(model, image), caption="Grad-CAM")

# ---------------- Tab 2: results ----------------
with tab_results:
    st.subheader("Validation vs. test (test set run once)")
    st.dataframe(RESULTS.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Test confusion matrix")
    cm = pd.DataFrame(
        [[TN, FP], [FN, TP]],
        index=["Actual: Normal", "Actual: Pneumonia"],
        columns=["Predicted: Normal", "Predicted: Pneumonia"],
    )
    st.table(cm)
    st.caption(f"TP {TP} | TN {TN} | FP {FP} | FN {FN}  (n = {TP + TN + FP + FN})")

    st.subheader("Baselines")
    n_test, n_pneu = TP + TN + FP + FN, TP + FN
    base = n_pneu / n_test
    st.write(
        f"The test set is {n_pneu}/{n_test} = **{base:.1%} pneumonia** (about 74% in train/val). "
        f"Always predicting 'pneumonia' would score **{base:.1%} accuracy** with 0% specificity, "
        "so accuracy alone is misleading here. Recall, specificity and AUC give the fuller picture."
    )

    st.subheader("Why is test worse than validation?")
    st.markdown(
        "- Specificity dropped from 96.3% to 66.2%, while recall stayed high and AUC stayed strong (0.983).\n"
        "- A high AUC with low specificity means the model **ranks cases well**, but the "
        "**decision threshold** is not well suited to the test distribution.\n"
        "- The model is trained **unweighted** on data that is about 74% pneumonia, so it leans toward "
        "predicting pneumonia. The test set has more normal cases (37.5%), which exposes this.\n"
        "- Results are reported as-is: the test set was run once and not used for retuning."
    )

# ---------------- Tab 3: settings & limitations ----------------
with tab_about:
    st.subheader("Frozen settings")
    st.table(pd.DataFrame({
        "Setting": ["Model", "Image size", "Class weights", "Learning rate", "Seed", "Epochs",
                    "Threshold", "Preprocessing", "Augmentation (training)", "Augmentation (at inference)"],
        "Value": [MODEL_NAME, f"{IMG_SIZE}x{IMG_SIZE}", CLASS_WEIGHTS, f"{LEARNING_RATE:g}", SEED,
                  f"{EPOCHS} (best validation AUC 0.9985)", THRESHOLD,
                  "Grayscale -> resize -> 3 channels -> normalize", TRAIN_AUGMENTATION, "None"],
    }))
    st.caption("Learning rate: 7 values tested from 1e-5 to 1e-2; 3e-5 gave the best validation results.")

    st.subheader("Limitations")
    st.markdown(
        "- Pediatric patients only, from a single source\n"
        "- Small, low-resolution images (64x64)\n"
        "- Many false positives on normal X-rays (specificity 66.2% on test)\n"
        "- Not validated for clinical use; educational demo only"
    )

st.divider()
st.caption(DISCLAIMER)
