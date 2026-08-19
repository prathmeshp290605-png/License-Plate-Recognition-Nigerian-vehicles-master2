import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 License Plate Recognition")
st.write("Upload a vehicle image to detect and read the license plate.")

uploaded_file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"]
)

def detect_plate(image):
    """
    Basic license-plate region detection.
    This version is independent of the old wxPython GUI,
    so it can run on Streamlit Cloud.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    plate = None
    box = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h else 0

            if 2.0 <= aspect_ratio <= 6.5 and w > 80 and h > 15:
                plate = image[y:y+h, x:x+w]
                box = (x, y, w, h)
                break

    return plate, box


if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read the uploaded image.")
    else:
        st.subheader("Uploaded Image")
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        if st.button("🔍 Detect License Plate", use_container_width=True):
            with st.spinner("Detecting license plate..."):
                plate, box = detect_plate(image)

            if plate is not None:
                x, y, w, h = box

                result = image.copy()
                cv2.rectangle(
                    result,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

                st.success("License plate region detected!")

                st.subheader("Detection Result")
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

                st.subheader("Detected Plate Region")
                st.image(
                    cv2.cvtColor(plate, cv2.COLOR_BGR2RGB),
                    use_container_width=True
                )

                st.info(
                    "The plate region has been detected. "
                    "For automatic plate-number OCR, connect your "
                    "existing OCR module to this Streamlit interface."
                )
            else:
                st.warning(
                    "No license plate region was detected. "
                    "Try a clearer vehicle image with the plate visible."
                )
else:
    st.info("Please upload a vehicle image to begin.")
