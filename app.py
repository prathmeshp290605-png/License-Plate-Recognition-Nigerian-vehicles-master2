import cv2
import easyocr
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🚗 License Plate Detection")

# EasyOCR Model Load
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

reader = load_ocr()

# File Uploader
uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect License Plate"):
        with st.spinner("Detecting license plate..."):
            results = reader.readtext(img_array)
            
            annotated = img_array.copy()
            h_img, w_img, _ = img_array.shape
            
            all_x1, all_y1, all_x2, all_y2 = [], [], [], []
            detected_texts = []

            # १. सर्व डिटेक्ट झालेल्या टेक्स्टचे कोऑर्डिनेट्स एकत्र करणे
            for (bbox, text, prob) in results:
                if prob > 0.20 and len(text.strip()) >= 2:
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    all_x1.append(int(top_left[0]))
                    all_y1.append(int(top_left[1]))
                    all_x2.append(int(bottom_right[0]))
                    all_y2.append(int(bottom_right[1]))
                    detected_texts.append(text)

            plate_crop = None
            full_text = " ".join(detected_texts)

            # २. संपूर्ण नंबर प्लेट एकत्र क्रॉप करणे (MH 12 HN 1375)
            if all_x1:
                x1 = min(all_x1)
                y1 = min(all_y1)
                x2 = max(all_x2)
                y2 = max(all_y2)

                # पॅडिंग (Padding)
                pad_x, pad_y = 25, 15
                x1_p, y1_p = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2_p, y2_p = min(w_img, x2 + pad_x), min(h_img, y2 + pad_y)

                plate_crop = img_array[y1_p:y2_p, x1_p:x2_p]
                cv2.rectangle(annotated, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 4)

            # ३. निकाल स्क्रीनवर दाखवणे
            st.subheader("Detection Result")
            st.image(annotated, caption="Detection Output", use_container_width=True)

            if plate_crop is not None:
                st.subheader("Detected Plate Region")
                st.image(plate_crop, caption=f"Detected Text: {full_text}", width=350)
                st.success(f"पूर्ण नंबर प्लेट यशस्वीपणे सापडली! ({full_text})")
            else:
                st.error("नंबर प्लेट सापडली नाही.")
