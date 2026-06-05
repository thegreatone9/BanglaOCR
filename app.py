import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import json
import os
import tensorflow as tf


# Page Configuration
st.set_page_config(
    page_title="Bangla OCR — Handwritten Character Reader",
    page_icon="🔤",
    layout="wide"
)


st.markdown("""
<style>
    /* Title centering */
    h1 { text-align: center; }

    /* Predicted word box — uses Streamlit's own vars, works in light and dark */
    .predicted-word {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        padding: 20px;
        background: var(--secondary-background-color);
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        margin: 8px 0 16px 0;
        letter-spacing: 4px;
    }
</style>
""", unsafe_allow_html=True)


# Model & Labels Loading (Cached)
@st.cache_resource
def load_model():
    """Load the trained Keras model from disk."""
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.keras")
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(model_path)
    return model


@st.cache_resource
def load_labels():
    """Load label mappings from labels.json."""
    labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
    if not os.path.exists(labels_path):
        return None
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels


# Segmentation — Matra-Aware 3-Phase Approach
def detect_and_remove_matra(binary_img):
    """
    Detect the Matra (horizontal headline) via horizontal projection profile
    and remove it to separate connected characters.

    Returns: (processed_image, matra_found: bool, matra_rows: tuple or None)
    """
    h, w = binary_img.shape

    # Horizontal projection: sum pixels per row
    h_proj = np.sum(binary_img, axis=1) / 255.0

    # Matra should be in the upper ~40% of the image
    upper_limit = int(h * 0.4)
    upper_region = h_proj[:upper_limit]

    if len(upper_region) == 0 or np.max(upper_region) == 0:
        return binary_img, False, None

    # Find rows where pixel density exceeds 60% of the peak in upper region
    peak = np.max(upper_region)
    threshold = peak * 0.6
    matra_rows = np.where(upper_region > threshold)[0]

    # Threshold to be a Matra
    if len(matra_rows) < 2:
        return binary_img, False, None

    matra_height = matra_rows[-1] - matra_rows[0] + 1

    # Matra shouldn't be more than 20% of image height
    if matra_height > h * 0.2:  
        return binary_img, False, None

    # Remove the Matra by zeroing those rows
    result = binary_img.copy()

    # Add a small margin around the matra rows
    start_row = max(0, matra_rows[0] - 1)
    end_row = min(h, matra_rows[-1] + 2)
    result[start_row:end_row, :] = 0

    return result, True, (matra_rows[0], matra_rows[-1])


def segment_by_vertical_projection(binary_img, min_char_width=10):
    """
    Use vertical projection profile to find natural gaps between characters
    after Matra removal.

    Returns: list of (x_start, x_end) tuples for each character segment
    """
    v_proj = np.sum(binary_img, axis=0) / 255.0

    if np.max(v_proj) == 0:
        return []

    # Eliminate noise with thresholding: consider columns with very low pixel counts as gaps
    threshold = np.max(v_proj) * 0.05
    is_gap = v_proj <= threshold

    segments = []
    in_char = False
    start = 0

    for col in range(len(is_gap)):
        if not is_gap[col] and not in_char:
            start = col
            in_char = True
        elif is_gap[col] and in_char:
            if col - start >= min_char_width:
                segments.append((start, col))
            in_char = False

    # Handle last character
    if in_char and (len(is_gap) - start >= min_char_width):
        segments.append((start, len(is_gap)))

    return segments


def merge_close_boxes(boxes, x_threshold=15):
    """Merge bounding boxes that are close together horizontally."""
    if not boxes:
        return []

    merged = [list(boxes[0])]
    
    for box in boxes[1:]:
        prev = merged[-1]
        prev_right = prev[0] + prev[2]

        if box[0] - prev_right < x_threshold:
            # Merge
            new_x = min(prev[0], box[0])
            new_y = min(prev[1], box[1])
            new_right = max(prev_right, box[0] + box[2])
            new_bottom = max(prev[1] + prev[3], box[1] + box[3])
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append(list(box))

    return [tuple(b) for b in merged]


def segment_by_contours(binary_img, min_area=100):
    """
    Fallback: use connected component analysis when no Matra is detected.

    Returns: list of (x, y, w, h) bounding boxes sorted left-to-right
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    # Merge close/overlapping boxes (for multi-stroke characters)
    merged = merge_close_boxes(boxes)

    return merged


def preprocess_char(char_img, target_size=64):
    """
    Pad the character image to a square, resize to target_size x target_size,
    and normalize to [0, 1].
    """
    h, w = char_img.shape[:2]

    # Pad to square
    max_dim = max(h, w)
    pad_h = (max_dim - h) // 2
    pad_w = (max_dim - w) // 2
    padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
    padded[pad_h:pad_h + h, pad_w:pad_w + w] = char_img

    # Add some border padding (10% on each side)
    border = int(max_dim * 0.1)
    bordered = np.zeros((max_dim + 2 * border, max_dim + 2 * border), dtype=np.uint8)
    bordered[border:border + max_dim, border:border + max_dim] = padded

    # Resize to target size
    resized = cv2.resize(bordered, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    return normalized


def segment_word(canvas_image):
    """
    Full segmentation pipeline:
    1. Convert to grayscale and binarize
    2. Try Matra detection + vertical projection
    3. Fall back to CCA if no Matra found
    4. Extract and preprocess character images
    """
    # Convert RGBA canvas to grayscale
    if len(canvas_image.shape) == 3 and canvas_image.shape[2] == 4:
        gray = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
    elif len(canvas_image.shape) == 3:
        gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = canvas_image.copy()

    # Binarize: ink becomes white (255), background becomes black (0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Check if canvas is empty
    if np.sum(binary) == 0:
        return [], binary

    # Phase 1: Try Matra detection and removal
    processed, matra_found, matra_info = detect_and_remove_matra(binary)

    char_images = []

    if matra_found:
        # Phase 2: Vertical projection segmentation
        segments = segment_by_vertical_projection(processed)

        if segments:
            for (x_start, x_end) in segments:
                # Crop from the ORIGINAL binary image (with full character strokes)
                char_region = binary[:, x_start:x_end]
                # Find vertical bounds (trim top/bottom whitespace)
                rows_with_content = np.where(np.sum(char_region, axis=1) > 0)[0]
                if len(rows_with_content) > 0:
                    y_start = rows_with_content[0]
                    y_end = rows_with_content[-1] + 1
                    char_crop = char_region[y_start:y_end, :]
                    char_images.append(preprocess_char(char_crop))
        else:
            # Vertical projection found no segments, fall back to CCA
            matra_found = False

    if not matra_found:
        # Phase 3: CCA fallback
        boxes = segment_by_contours(binary)
        for (x, y, w, h) in boxes:
            char_crop = binary[y:y + h, x:x + w]
            char_images.append(preprocess_char(char_crop))

    return char_images, binary


def predict_characters(model, char_images, labels):
    """Predict each character and return results."""
    # Determine if model expects 1 or 3 channels
    input_channels = model.input_shape[-1]

    results = []
    for char_img in char_images:
        if input_channels == 3:
            # Convert grayscale to RGB by stacking
            rgb_img = np.stack([char_img] * 3, axis=-1)
            input_img = rgb_img.reshape(1, 64, 64, 3)
        else:
            # Single channel (CNN)
            input_img = char_img.reshape(1, 64, 64, 1)

        # Predict
        predictions = model.predict(input_img, verbose=0)

        # Top-3 predictions
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = [(labels.get(str(i), f"class_{i}"), float(predictions[0][i])) for i in top3_indices]

        results.append({
            "image": char_img,
            "top_prediction": top3[0],
            "top3": top3,
        })

    return results


# UI Content
st.title("🔤 Bangla OCR")

model = load_model()
labels = load_labels()

model_ok = True
if model is None:
    st.error(
        "🚫 **Model not found!** Could not locate `models/model.keras`. "
        "Please run `python train.py` first to train and save the model."
    )
    model_ok = False

if labels is None:
    st.error(
        "🚫 **Labels not found!** Could not locate `labels.json`. "
        "Please ensure the labels file exists in the project root."
    )
    model_ok = False

#Two-column layout
left_col, right_col = st.columns([3, 3], gap="large")

with left_col:
    st.subheader("✍️ Drawing Canvas")
    st.info("Draw a Bangla word or character. Use the toolbar above the canvas to switch tools or clear (🗑️).")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=8,
        stroke_color="#000000",
        background_color="#FFFFFF",
        width=620,
        height=220,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Predict button
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)


with right_col:
    st.subheader("📊 Prediction Results")

    if predict_clicked:
        if not model_ok:
            st.warning("⚠️ Cannot predict — model or labels are not loaded. See errors above.")
        elif canvas_result.image_data is None:
            st.warning("⚠️ The canvas is empty. Please draw something first!")
        else:
            canvas_image = canvas_result.image_data.astype(np.uint8)

            # Quick check: is the canvas actually blank?
            if len(canvas_image.shape) == 3 and canvas_image.shape[2] == 4:
                check_gray = cv2.cvtColor(canvas_image, cv2.COLOR_RGBA2GRAY)
            elif len(canvas_image.shape) == 3:
                check_gray = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
            else:
                check_gray = canvas_image
            _, check_bin = cv2.threshold(check_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            if np.sum(check_bin) == 0:
                st.warning("⚠️ The canvas appears to be empty. Please draw a character or word!")
            else:
                with st.spinner("Segmenting & predicting…"):
                    char_images, binary_img = segment_word(canvas_image)

                if not char_images:
                    st.warning("⚠️ No characters could be segmented from the drawing. Try drawing more clearly.")
                else:
                    results = predict_characters(model, char_images, labels)

                    # Build the combined word
                    predicted_word = "".join([r["top_prediction"][0] for r in results])

                    # Display combined word
                    st.markdown("**Predicted Word**")
                    st.markdown(f'<div class="predicted-word">{predicted_word}</div>', unsafe_allow_html=True)

                    # Display individual character results
                    st.markdown(f"**Detected {len(results)} character(s)**")

                    # Create columns for character cards
                    num_chars = len(results)
                    cols_per_row = min(num_chars, 4)

                    for row_start in range(0, num_chars, cols_per_row):
                        row_end = min(row_start + cols_per_row, num_chars)
                        char_cols = st.columns(row_end - row_start)

                        for idx, col in enumerate(char_cols):
                            r = results[row_start + idx]

                            with col:
                                # Show the segmented character image
                                display_img = (r["image"] * 255).astype(np.uint8)
                                st.image(display_img, width=80, caption=f"Char {row_start + idx + 1}")

                                # Top prediction
                                label, conf = r["top_prediction"]
                                st.markdown(f"### {label}")
                                st.caption(f"{conf * 100:.1f}%")

                                # Top-3 alternatives
                                alts = " · ".join(
                                    [f"{lbl} ({c * 100:.0f}%)" for lbl, c in r["top3"][1:]]
                                )
                                if alts:
                                    st.caption(f"Alt: {alts}")

                    # ── Show segmented characters in a row below the canvas ──
                    with left_col:
                        st.subheader("🧩 Segmented Characters")
                        seg_cols = st.columns(min(len(results), 8))

                        for i, r in enumerate(results):
                            with seg_cols[i % len(seg_cols)]:
                                display_img = (r["image"] * 255).astype(np.uint8)
                                st.image(display_img, width=64, caption=r["top_prediction"][0])
    else:
        st.write("✨ Draw on the canvas and click **Predict** to see results here.")
