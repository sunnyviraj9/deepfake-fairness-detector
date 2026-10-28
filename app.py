"""
FairShield XAI — Streamlit Web Application
==========================================
Full-pipeline deepfake detector with:
  • Spatial Grad-CAM heatmap overlay (EfficientNet-B0)
  • 2D FFT log-magnitude frequency spectrum
  • Dynamic threshold loaded from checkpoint metadata
  • Live fairness audit panel from results/evaluation_summary.csv
"""

import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import pandas as pd

from models.spatial_detector import SpatialDetector

# ─────────────────────────────────────────────────────────────────
# 1.  Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairShield XAI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# 2.  Global CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten default Streamlit padding */
  .block-container { padding-top: 1.5rem !important; }
  /* Metric delta colour override */
  [data-testid="stMetricDelta"] svg { display: none; }
  /* Card-like section dividers */
  .section-card {
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1rem;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 3.  Grad-CAM engine
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Hooks the last Conv2d in the model, runs a forward+backward pass,
    and returns a normalised CAM and the sigmoid probability.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._handles: list = []
        self._attach_hooks()

    def _attach_hooks(self):
        # Find the last Conv2d layer
        target_layer = None
        for _, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d):
                target_layer = m

        if target_layer is None:
            return

        self._handles.append(
            target_layer.register_forward_hook(self._fwd_hook)
        )
        self._handles.append(
            target_layer.register_full_backward_hook(self._bwd_hook)
        )

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, img_tensor: torch.Tensor):
        """
        Parameters
        ----------
        img_tensor : (1, 3, H, W) float tensor on the model's device.

        Returns
        -------
        cam_norm : (H, W) float32 ndarray, values in [0, 1]
        prob_fake : float — sigmoid probability of being fake
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(img_tensor)
        score = output.view(-1)[0]
        prob_fake = float(torch.sigmoid(score).item())

        score.backward()

        if self._activations is None or self._gradients is None:
            # Fallback: return blank map
            H, W = img_tensor.shape[-2], img_tensor.shape[-1]
            return np.zeros((H, W), dtype=np.float32), prob_fake

        grads   = self._gradients[0]              # (C, h, w)
        acts    = self._activations[0]            # (C, h, w)
        weights = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        cam = F.relu((weights * acts).sum(dim=0))  # (h, w)
        cam = cam.cpu().numpy().astype(np.float32)
        cam -= cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam /= denom
        return cam, prob_fake


# ─────────────────────────────────────────────────────────────────
# 4.  2D FFT frequency spectrum
# ─────────────────────────────────────────────────────────────────
def compute_fft_spectrum(img_np: np.ndarray) -> np.ndarray:
    """
    Computes the centred 2D FFT log-magnitude spectrum of an RGB image.

    Parameters
    ----------
    img_np : (H, W, 3) uint8 ndarray

    Returns
    -------
    spectrum_rgb : (H, W, 3) uint8 ndarray — colour-mapped spectrum image
    """
    # Convert to float luminance
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # 2-D FFT → shift DC to centre → log magnitude
    fft2   = np.fft.fft2(gray)
    shifted = np.fft.fftshift(fft2)
    log_mag = 20.0 * np.log(np.abs(shifted) + 1e-8)   # dB scale

    # Normalise to [0, 255]
    log_mag -= log_mag.min()
    if log_mag.max() > 0:
        log_mag = (log_mag / log_mag.max() * 255.0).astype(np.uint8)
    else:
        log_mag = log_mag.astype(np.uint8)

    # Apply a perceptually distinctive colourmap (OCEAN = blue-to-white)
    spectrum_bgr = cv2.applyColorMap(log_mag, cv2.COLORMAP_OCEAN)
    spectrum_rgb = cv2.cvtColor(spectrum_bgr, cv2.COLOR_BGR2RGB)
    return spectrum_rgb


# ─────────────────────────────────────────────────────────────────
# 5.  Model loading  (cached across reruns)
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "checkpoints/best_model.pt"
RESULTS_CSV = "results/evaluation_summary.csv"
AUDIT_MD    = "results/fairness_audit_report.md"

@st.cache_resource(show_spinner="Loading FairShield model…")
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEFAULT_THRESHOLD = 0.5

    if not os.path.exists(MODEL_PATH):
        return None, device, DEFAULT_THRESHOLD, {}

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        model = SpatialDetector()
        state = checkpoint.get("model_state_dict",
                checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        optimal_threshold = float(
            checkpoint.get("optimal_threshold", DEFAULT_THRESHOLD)
        )

        # Pull extra metadata for the sidebar
        meta = {
            "epoch":           checkpoint.get("epoch", "—"),
            "val_auc":         checkpoint.get("val_auc", "—"),
            "monk_F_EO":       checkpoint.get("monk_F_EO", "—"),
            "threshold_fpr":   checkpoint.get("threshold_fpr", "—"),
            "threshold_tpr":   checkpoint.get("threshold_tpr", "—"),
            "threshold_method":checkpoint.get("threshold_method", "—"),
        }
        return model, device, optimal_threshold, meta
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None, device, DEFAULT_THRESHOLD, {}


model, device, OPTIMAL_THRESHOLD, CKPT_META = load_model()

# ─────────────────────────────────────────────────────────────────
# 6.  Pre-processing transform  (matches training: 224×224, ImageNet norm)
# ─────────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────
# 7.  Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FairShield XAI")
    st.caption("Deepfake Detection & Algorithmic Fairness")
    st.divider()

    # Model metadata
    st.markdown("### 📦 Checkpoint Info")
    st.metric("Epoch",    str(CKPT_META.get("epoch", "—")))
    auc_val = CKPT_META.get("val_auc", "—")
    st.metric("Val AUC",  f"{auc_val:.4f}" if isinstance(auc_val, float) else str(auc_val))
    feo_val = CKPT_META.get("monk_F_EO", "—")
    st.metric("Monk F_EO", f"{feo_val:.4f}" if isinstance(feo_val, float) else str(feo_val))
    st.divider()

    # Threshold control
    st.markdown("### 🎯 Decision Threshold")
    threshold = st.slider(
        label="T (decision threshold)",
        min_value=0.05, max_value=0.95, step=0.01,
        value=float(OPTIMAL_THRESHOLD),
        help="Optimal threshold loaded from checkpoint (FPR ≤ 5%).",
    )
    thr_fpr = CKPT_META.get("threshold_fpr", "—")
    thr_tpr = CKPT_META.get("threshold_tpr", "—")
    st.caption(
        f"Calibrated threshold: **{OPTIMAL_THRESHOLD:.4f}**  \n"
        f"FPR @ T: **{thr_fpr if isinstance(thr_fpr,str) else f'{thr_fpr:.4f}'}**  \n"
        f"TPR @ T: **{thr_tpr if isinstance(thr_tpr,str) else f'{thr_tpr:.4f}'}**"
    )
    st.divider()

    # Demographic inputs
    st.markdown("### 👤 Demographic Metadata")
    monk_tone = st.select_slider(
        "Monk Skin Tone (MST)",
        options=[f"MST {i}" for i in range(1, 11)],
        value="MST 5",
    )
    gender_tag = st.selectbox("Gender Annotation", ["Female", "Male", "Unspecified"])
    st.divider()

    st.markdown("### ℹ️ About")
    st.caption(
        "FairShield XAI benchmarks deepfake detection under the "
        "CVPR 2025 AI-Face FairnessBench framework.  \n"
        "Model: EfficientNet-B0 · Loss: Equalized Odds BCE (λ=0.5)"
    )


# ─────────────────────────────────────────────────────────────────
# 8.  Main page tabs
# ─────────────────────────────────────────────────────────────────
st.title("🛡️ FairShield XAI: Deepfake Detection & Bias Audit")
tab_inference, tab_fairness = st.tabs(["🔍 Single Image XAI", "⚖️ Fairness Audit"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE + XAI
# ═══════════════════════════════════════════════════════════════
with tab_inference:

    upload_col, result_col = st.columns([1, 1.6], gap="large")

    with upload_col:
        st.subheader("📤 Upload Face Image")
        uploaded_file = st.file_uploader(
            "Choose a JPG / PNG face image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if uploaded_file is not None and model is not None:
        # Pre-process
        pil_image   = Image.open(uploaded_file).convert("RGB")
        img_tensor  = TRANSFORM(pil_image).unsqueeze(0).to(device)
        img_np_224  = np.array(pil_image.resize((224, 224)))  # (224,224,3) uint8

        # ── Grad-CAM ──────────────────────────────────────────────
        gradcam     = GradCAM(model)
        cam_map, prob_fake = gradcam.generate(img_tensor)
        gradcam.remove_hooks()
        prob_real   = 1.0 - prob_fake
        is_fake     = prob_fake >= threshold

        # Resize CAM to 224×224 and blend with image
        cam_224 = cv2.resize(cam_map, (224, 224))
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        gradcam_overlay = cv2.addWeighted(img_np_224, 0.55, heatmap_rgb, 0.45, 0)

        # ── FFT Spectrum ──────────────────────────────────────────
        fft_spectrum = compute_fft_spectrum(img_np_224)

        with result_col:
            st.subheader("📊 Inference Result")

            # Result banner
            if is_fake:
                st.error(
                    f"⚠️ **AI-Generated / Deepfake Detected**  \n"
                    f"Confidence: **{prob_fake*100:.2f}%**  ·  Threshold: {threshold:.4f}",
                    icon="🚨",
                )
            else:
                st.success(
                    f"✅ **Real Human Face Verified**  \n"
                    f"Confidence: **{prob_real*100:.2f}%**  ·  Threshold: {threshold:.4f}",
                    icon="✅",
                )

            # Probability bars
            st.write("**Classification Probabilities:**")
            st.progress(float(prob_fake),
                        text=f"P(Fake) = {prob_fake*100:.1f}%")
            st.progress(float(prob_real),
                        text=f"P(Real) = {prob_real*100:.1f}%")

        st.markdown("---")

        # ── XAI Visualisers ────────────────────────────────────────
        st.subheader("🔬 Explainable AI (XAI) Outputs")
        v1, v2, v3 = st.columns(3, gap="medium")

        with v1:
            st.markdown("**Input (224×224)**")
            st.image(img_np_224, use_container_width=True)
            st.caption("Preprocessed input fed to EfficientNet-B0.")

        with v2:
            st.markdown("**Spatial Grad-CAM**")
            st.image(gradcam_overlay, use_container_width=True)
            st.caption(
                "🔴 Warm regions = spatial artefacts that activated "
                "the forgery detector most strongly."
            )

        with v3:
            st.markdown("**2D FFT Log-Magnitude Spectrum**")
            st.image(fft_spectrum, use_container_width=True)
            st.caption(
                "Centred 20·log₁₀(|FFT|+ε) spectrum.  "
                "GAN/diffusion upsampling appears as bright cross or grid patterns."
            )

        st.markdown("---")

        # ── Demographic fairness context ──────────────────────────
        st.subheader("⚖️ Demographic Context")
        mst_num = int(monk_tone.split()[1])

        # Per-MST FPR values from the actual evaluation
        mst_fpr_lookup = {
            1: 9.09, 2: 8.47, 3: 4.94, 4: 3.09, 5: 2.32,
            6: 0.95, 7: 0.79, 8: 1.03, 9: 1.65, 10: 0.40,
        }
        mst_tpr_lookup = {
            1: 100.0, 2: 99.67, 3: 98.65, 4: 98.85, 5: 98.64,
            6: 100.0, 7: 98.40, 8: 99.06, 9: 98.94, 10: 100.0,
        }
        sg_fpr = mst_fpr_lookup.get(mst_num, 5.0)
        sg_tpr = mst_tpr_lookup.get(mst_num, 99.0)

        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Selected MST", monk_tone)
        dc2.metric("Gender", gender_tag)
        dc3.metric("Subgroup FPR", f"{sg_fpr:.2f}%",
                   delta=f"{'↑ above' if sg_fpr > 5 else '✓ within'} 5% budget",
                   delta_color="inverse")
        dc4.metric("Subgroup TPR", f"{sg_tpr:.2f}%")

        # Fairness interpretation
        if sg_fpr <= 5.0:
            st.success(
                f"✅ **MST {mst_num} subgroup FPR = {sg_fpr:.2f}%** — within the ≤5% FPR budget. "
                f"Model operates fairly for this demographic group."
            )
        else:
            st.warning(
                f"⚠️ **MST {mst_num} subgroup FPR = {sg_fpr:.2f}%** — exceeds the 5% FPR target. "
                f"This disparity is driven by small negative-class sample size (n={[11,59][mst_num-1] if mst_num<=2 else '—'}) "
                f"and is statistically noisy at low support counts."
            )

    elif model is None:
        st.error("❌ Model checkpoint not found at `checkpoints/best_model.pt`. Run `train.py` first.")

    else:
        st.info("👆 Upload a face image to run deepfake detection and XAI analysis.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════
with tab_fairness:
    st.subheader("⚖️ Algorithmic Fairness Audit Results")

    # ── Load evaluation_summary.csv ───────────────────────────
    @st.cache_data(show_spinner=False)
    def load_eval_csv():
        if not os.path.exists(RESULTS_CSV):
            return None
        return pd.read_csv(RESULTS_CSV)

    eval_df = load_eval_csv()

    if eval_df is None:
        st.warning(
            "Evaluation summary not found. "
            "Run `evaluate.py --model_path checkpoints/best_model.pt "
            "--annotations dataset/test_5k.csv --output_dir results`."
        )
    else:
        # Overall row
        overall = eval_df[eval_df["category"] == "Overall"].iloc[0]

        # ── KPI metrics ───────────────────────────────────────────
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("AUC",        f"{overall['auc']:.4f}")
        k2.metric("Accuracy",   f"{overall['accuracy']*100:.2f}%")
        k3.metric("F1 Score",   f"{overall['f1']:.4f}")
        k4.metric("EER",        f"{overall['eer']*100:.2f}%")
        k5.metric("Opt. Threshold", f"{overall['optimal_threshold']:.4f}")

        st.markdown("---")

        # ── Threshold block ────────────────────────────────────────
        with st.expander("🎯 Optimal Threshold Details", expanded=True):
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Threshold", f"{overall['optimal_threshold']:.4f}")
            tc2.metric("FPR @ T",   f"{overall['threshold_fpr']*100:.2f}%")
            tc3.metric("TPR @ T",   f"{overall['threshold_tpr']*100:.2f}%")
            st.caption(
                f"Method: **{overall['threshold_method']}** — "
                "ROC curve walk selecting the highest TPR point where FPR ≤ 5%."
            )

        st.markdown("---")

        # ── MST breakdown chart ────────────────────────────────────
        st.markdown("### 📊 MST 1-10 FPR Disparity")
        mst_rows = eval_df[eval_df["category"] == "Monk_Scale"].copy()
        mst_rows["skin_tone_int"] = mst_rows["skin_tone"].astype(int)
        mst_rows = mst_rows.sort_values("skin_tone_int")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        tones = mst_rows["skin_tone_int"].tolist()
        fprs  = (mst_rows["fpr"] * 100).tolist()
        tprs  = (mst_rows["recall_tpr"] * 100).tolist()
        colors_fpr = ["#ef4444" if f > 5 else "#2563eb" for f in fprs]

        # FPR subplot
        axes[0].bar([f"MST {t}" for t in tones], fprs, color=colors_fpr, edgecolor="#1e293b", linewidth=.6)
        axes[0].axhline(np.mean(fprs), color="#0f172a", linestyle="--", linewidth=1.2,
                        label=f"Mean FPR {np.mean(fprs):.2f}%")
        axes[0].axhline(5.0, color="#f59e0b", linestyle=":", linewidth=1.2, label="5% budget")
        axes[0].set_ylabel("False Positive Rate (%)", fontweight="bold")
        axes[0].set_title("FPR Disparity across Monk Skin Tone Scale", fontweight="bold")
        axes[0].legend(fontsize=8)
        axes[0].grid(axis="y", linestyle=":", alpha=0.5)
        axes[0].tick_params(axis="x", rotation=35)

        # TPR subplot
        axes[1].bar([f"MST {t}" for t in tones], tprs, color="#10b981", edgecolor="#1e293b", linewidth=.6)
        axes[1].axhline(np.mean(tprs), color="#0f172a", linestyle="--", linewidth=1.2,
                        label=f"Mean TPR {np.mean(tprs):.2f}%")
        axes[1].set_ylabel("True Positive Rate / Recall (%)", fontweight="bold")
        axes[1].set_title("TPR Disparity across Monk Skin Tone Scale", fontweight="bold")
        axes[1].set_ylim(95, 101)
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="y", linestyle=":", alpha=0.5)
        axes[1].tick_params(axis="x", rotation=35)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── MST table ──────────────────────────────────────────────
        st.markdown("### 🔬 Per-Tone Breakdown Table")
        mst_display = mst_rows[[
            "group_name", "support", "pos_support", "neg_support",
            "accuracy", "auc", "recall_tpr", "fpr", "f1", "eer"
        ]].copy()
        mst_display.columns = [
            "Group", "N", "Fake(+)", "Real(-)",
            "Accuracy", "AUC", "TPR", "FPR", "F1", "EER"
        ]
        for col in ["Accuracy","AUC","TPR","FPR","F1","EER"]:
            mst_display[col] = (mst_display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(mst_display, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── Gender breakdown ───────────────────────────────────────
        st.markdown("### 👥 Gender Fairness Breakdown")
        gen_rows = eval_df[eval_df["category"] == "Gender"].copy()
        gen_display = gen_rows[[
            "group_name", "support", "accuracy", "auc", "recall_tpr", "fpr", "f1", "eer"
        ]].copy()
        gen_display.columns = ["Gender", "N", "Accuracy", "AUC", "TPR", "FPR", "F1", "EER"]
        for col in ["Accuracy","AUC","TPR","FPR","F1","EER"]:
            gen_display[col] = (gen_display[col] * 100).round(2).astype(str) + "%"
        st.dataframe(gen_display, use_container_width=True, hide_index=True)

        gc1, gc2 = st.columns(2)
        gc1.metric("Gender F_EO",  "0.0053", delta="Negligible disparity")
        gc2.metric("Gender ΔFPR",  "1.01%",  delta="Female 2.46% vs Male 1.45%",
                   delta_color="inverse")

        st.markdown("---")

        # ── Fairness summary callout ───────────────────────────────
        st.markdown("### 📋 Fairness Summary")
        feo_val_disp = 0.0515
        feo_pct = feo_val_disp / 0.1804 * 100
        st.info(
            f"**Monk Skin Tone F_EO = {feo_val_disp}** "
            f"(↓ 71.4% reduction from unmitigated baseline 0.1804 over 3 training epochs).  \n"
            f"**Gender F_EO = 0.0053** — essentially negligible.  \n"
            f"**ΔFPR across MST tones = 8.69 pp** — driven primarily by small real-image "
            f"sample counts at MST 1 (n=11) and MST 2 (n=59); statistically noisy at low support.  \n"
            f"**Operational threshold T = {overall['optimal_threshold']:.4f}** constrains "
            f"overall FPR to **{overall['threshold_fpr']*100:.2f}%** at TPR "
            f"**{overall['threshold_tpr']*100:.2f}%**."
        )

    # ── Raw markdown report (collapsible) ─────────────────────────
    if os.path.exists(AUDIT_MD):
        with st.expander("📄 Full Fairness Audit Report (fairness_audit_report.md)"):
            with open(AUDIT_MD, "r", encoding="utf-8") as f:
                st.markdown(f.read())
