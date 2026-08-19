import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🚗 License Plate Detection")

# EasyOCR Model Cache (ॲप फास्ट चालण्यासाठी)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect License Plate"):
        with st.spinner("Detecting license plate..."):
            # EasyOCR द्वारे फोटोतील टेक्स्ट आणि लोकेशन शोधणे
            results = reader.readtext(img_array)
            
            annotated = img_array.copy()
            plate_crop = None
            detected_text = ""

            for (bbox, text, prob) in results:
                # जर टेक्स्ट ३ पेक्षा जास्त अक्षरांचे असेल आणि कॉन्फिडन्स योग्य असेल
                if prob > 0.25 and len(text.strip()) >= 4:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    x1, y1 = int(top_left[0]), int(top_left[1])
                    x2, y2 = int(bottom_right[0]), int(bottom_right[1])

                    # प्लेटच्या आकारासाठी थोडे पॅडिंग (Padding)
                    h_img, w_img, _ = img_array.shape
                    pad_x, pad_y = 15, 10
                    x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
                    x2_p, y2_p = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

                    plate_crop = img_array[y1_p:y2_p, x1_p:x2_p]
                    cv2.rectangle(annotated, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 4)
                    detected_text = text
                    break

            st.subheader("Detection Result")
            st.image(annotated, caption="Detection Output", use_container_width=True)

            if plate_crop is not None:
                st.subheader("Detected Plate Region")
                st.image(plate_crop, caption=f"Detected Text: {detected_text}", width=300)
                st.success(f"नंबर प्लेट यशस्वीपणे सापडली! (नंबर: {detected_text})")
            else:
                st.error("नंबर प्लेट सापडली नाही.")
