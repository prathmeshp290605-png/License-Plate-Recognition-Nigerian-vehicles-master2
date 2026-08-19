import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🚗 License Plate Detection")

# 1. Image Uploader
uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

# 2. Plate Detection Function
def detect_plate(img_array):
    h, w, _ = img_array.shape
    
    # Image Preprocessing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 100, 200)

    # Find Contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plate_crop = None
    best_box = None

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bw / float(bh)
        area = bw * bh

        # Filter 1: Number Plate Aspect Ratio (2.0 to 5.5)
        # Filter 2: Ignore top sky & extreme bottom ground (y between 15% and 75% of image height)
        # Filter 3: Minimum Area
        if 2.0 <= aspect_ratio <= 5.5 and (area > 1200) and (h * 0.15 < y < h * 0.75):
            best_box = (x, y, bw, bh)
            plate_crop = img_array[y:y+bh, x:x+bw]
            break

    annotated = img_array.copy()
    if best_box is not None:
        x, y, bw, bh = best_box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
        return annotated, plate_crop
    
    return annotated, None

# 3. Main Streamlit App Logic
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect License Plate"):
        with st.spinner("Detecting license plate..."):
            annotated_img, plate_crop = detect_plate(img_array)
            
            st.subheader("Detection Result")
            st.image(annotated_img, caption="Detected Vehicle", use_container_width=True)
            
            if plate_crop is not None:
                st.subheader("Detected Plate Region")
                st.image(plate_crop, caption="Cropped License Plate", width=300)
                st.success("नंबर प्लेट यशस्वीपणे सापडली!")
            else:
                st.error("नंबर प्लेट सापडली नाही. दुसरी इमेज वापरून पहा.")
