from datetime import datetime
import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="License Plate Recognition System",
    layout="wide",
    page_icon="🚗",
)

st.title("🚗 License Plate Recognition System")

# Session State for Detection History
if "history" not in st.session_state:
    st.session_state["history"] = []


# EasyOCR Model Load
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"])


reader = load_ocr()

# Sidebar - History & Export
st.sidebar.header("📊 Detection Logs")

# Main Interface Layout
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload vehicle image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        detect_btn = st.button("Detect License Plate", type="primary")

with col2:
    if uploaded_file is not None and detect_btn:
        with st.spinner("Detecting license plate..."):
            results = reader.readtext(img_array)

            annotated = img_array.copy()
            h_img, w_img, _ = img_array.shape

            all_x1, all_y1, all_x2, all_y2 = [], [], [], []
            detected_texts = []

            for bbox, text, prob in results:
                if prob > 0.20 and len(text.strip()) >= 2:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    all_x1.append(int(top_left[0]))
                    all_y1.append(int(top_left[1]))
                    all_x2.append(int(bottom_right[0]))
                    all_y2.append(int(bottom_right[1]))
                    detected_texts.append(text)

            plate_crop = None
            full_text = (
                " ".join(detected_texts).upper().replace(".", "").replace("-", "")
            )

            if all_x1:
                x1, y1 = min(all_x1), min(all_y1)
                x2, y2 = max(all_x2), max(all_y2)

                pad_x, pad_y = 25, 15
                x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2_p, y2_p = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

                plate_crop = img_array[y1_p:y2_p, x1_p:x2_p]
                cv2.rectangle(
                    annotated, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 4
                )

                # Add to History Log with Date & Time
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["history"].append(
                    {"Timestamp": now, "License Plate": full_text}
                )

            st.subheader("Results")
            st.image(
                annotated, caption="Detection Output", use_container_width=True
            )

            if plate_crop is not None:
                st.subheader("Cropped License Plate")
                st.image(
                    plate_crop, caption=f"Detected: {full_text}", width=320
                )
                st.success(f"नवी नोंद सेव्ह झाली: **{full_text}**")
            else:
                st.error("नंबर प्लेट सापडली नाही.")

# Sidebar History Display and CSV Download Button
if st.session_state["history"]:
    df = pd.DataFrame(st.session_state["history"])
    st.sidebar.dataframe(df, use_container_width=True)

    # Convert DataFrame to CSV for Download
    csv = df.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Download CSV Log",
        data=csv,
        file_name=f"license_plates_{datetime.now().strftime('%Y%m%m')}.csv",
        mime="text/csv",
    )
