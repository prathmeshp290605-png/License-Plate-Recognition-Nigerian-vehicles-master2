import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="License Plate Detector", layout="centered")
st.title("🚗 License Plate Detection")

uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

def detect_plate(img_array):
    h, w, _ = img_array.shape
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 1. Edge-preserving filter
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # 2. Sobel Vertical Gradient (अक्षरांच्या उभ्या रेषा शोधण्यासाठी)
    sobelx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
    
    # 3. Otsu Thresholding
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Morphological Close (अक्षरांचे छोटे बॉक्सेस एकत्र जोडून एक प्लेट बनवणे)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 5. Contours शोधणे
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plate_crop = None
    best_box = None

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bw / float(bh)
        area = bw * bh

        # नंबर प्लेटचे फिल्टर्स (Aspect ratio आणि Vertical Position)
        if 2.0 <= aspect_ratio <= 6.5 and area > 600 and (h * 0.20 < y < h * 0.80):
            best_box = (x, y, bw, bh)
            plate_crop = img_array[y:y+bh, x:x+bw]
            break

    annotated = img_array.copy()
    if best_box is not None:
        x, y, bw, bh = best_box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 4)
        return annotated, plate_crop
    
    return annotated, None

# Main Streamlit App Logic
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
                st.error("नंबर प्लेट सापडली नाही. थ्रेशोल्ड अडजस्ट करा किंवा EasyOCR वापरा.")
