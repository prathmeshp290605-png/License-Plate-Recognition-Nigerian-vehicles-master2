import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🚗 License Plate Detection")

uploaded_file = st.file_uploader("इमेज अपलोड करा...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Load Image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    h, w, _ = img_array.shape

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect License Plate"):
        # 2. Convert to Grayscale & Thresholding (OpenCV Image Processing)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blur, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_crop = None
        best_box = None

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            area = bw * bh

            # Filter 1: Aspect Ratio range for Indian Plates (2.5 to 5.5)
            # Filter 2: Ignore top sky region and bottom ground region (Focus on middle vertical region)
            # Filter 3: Minimum area requirement
            if 2.2 <= aspect_ratio <= 5.5 and (area > 1500) and (y > h * 0.30 and y < h * 0.85):
                best_box = (x, y, bw, bh)
                plate_crop = img_array[y:y+bh, x:x+bw]
                break

        annotated = img_array.copy()

        st.subheader("Detection Result")
        
        if best_box is not None:
            x, y, bw, bh = best_box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
            st.image(annotated, caption="Correct Plate Location Detected", use_column_width=True)
            
            st.subheader("Detected Plate Region")
            st.image(plate_crop, caption="Cropped Plate", width=300)
            st.success("नंबर प्लेट योग्य पद्धतीने सापडली आहे!")
        else:
            st.warning("ऑटोमॅटिक डिटेक्शन अयशस्वी. कृपया थ्रेशोल्ड किंवा इमेज तपासा.")
