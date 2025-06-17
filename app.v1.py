import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os
import json

# ------------------------------
# Configuration and Helper Functions
# ------------------------------

# Set up a config file for celebrity databases (can be replaced with database or JSON)
CELEBRITY_DATABASES = {
    "default": ["Tom_Cruise.jpg", "Brad_Pitt.jpg", "Leonardo_DiCaprio.jpg"],
    "custom": []  # Will be populated from uploaded files
}

# Load celebrity encodings once and cache them


def load_celebrity_encodings(celeb_images):
    st.info("Loading celebrity encodings...")
    celebrity_encodings = []

    for image_path in celeb_images:
        # Extract the base filename and remove .jpg extension
        name_without_extension = image_path.split('.')[0]

        # Load the celebrity image and get face encoding
        try:
            celeb_image = face_recognition.load_image_file(image_path)
            # Ensure there is at least one face in the image
            if len(face_recognition.face_locations(celeb_image)) == 0:
                st.warning(f"No face found in {image_path}. Skipping.")
                continue
            celeb_encoding = face_recognition.face_encodings(celeb_image)[0]
            celebrity_encodings.append(
                (name_without_extension, celeb_encoding))
        except Exception as e:
            st.error(f"Error loading {image_path}: {str(e)}")
            continue

    return celebrity_encodings


def detect_celebrities(img, celeb_encodings):
    # Convert PIL image to RGB (required for face_recognition)
    img_rgb = np.array(img)

    # Find all the faces and face encodings in the current frame of video
    try:
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(
            img_rgb, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Find the best match
            matches = []
            name = "Unknown"
            confidence = 0.0

            for celeb_name, celeb_encoding in celeb_encodings:
                # Compare face encoding
                match = face_recognition.compare_faces(
                    [celeb_encoding], face_encoding)
                distance = face_recognition.face_distance(
                    [celeb_encoding], face_encoding)

                # Use a confidence threshold (e.g., 0.6)
                if distance < 0.6:
                    name = celeb_name
                    confidence = 1 - distance

            # Draw a box around the face
            cv2.rectangle(img_rgb, (left, top),
                          (right, bottom), (0, 0, 255), 2)

            # Calculate text size to ensure it fits within available space
            font_scale = min(1.0, ((bottom - top) / 40))
            thickness = max(1, int(font_scale * 3))

            # Get the dimensions of the text box
            (text_width, text_height), _ = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Calculate bottom_text position with a fixed margin below the face rectangle
            margin = 10  # Adjust this value to increase or decrease the distance between the bounding box and the name

            if (bottom + text_height + margin) < img_rgb.shape[0]:
                # Draw a background box for the text
                cv2.rectangle(img_rgb, (left, bottom),
                              (left + text_width, bottom + text_height + margin),
                              (0, 0, 255), thickness)

                # Write the name on top of the rectangle
                cv2.putText(img_rgb, f"{name} ({confidence:.2f})", (left, bottom + text_height + margin),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    except Exception as e:
        st.error(f"Error detecting celebrities: {str(e)}")
        pass

    # Convert back to PIL image
    img_pil = Image.fromarray(img_rgb)

    return img_pil


# ------------------------------
# Streamlit App
# ------------------------------
st.title("Celebrity Face Recognition")
st.write("Upload an image to detect any known celebrities using multiple models.")

# Select celebrity database
selected_db = st.selectbox("Select Celebrity Database",
                           list(CELEBRITY_DATABASES.keys()))

# Load celebrity encodings based on selected database
celeb_images = CELEBRITY_DATABASES[selected_db]
if selected_db == "custom":
    uploaded_celebs = st.file_uploader(
        "Upload celebrity images", type="jpg", accept_multiple_files=True)
    if uploaded_celebs:
        for file in uploaded_celebs:
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
        celeb_images = [f.name for f in uploaded_celebs]
else:
    # Use hardcoded celebrity images
    celeb_images = CELEBRITY_DATABASES[selected_db]

# Load encodings once and cache
celeb_encodings = load_celebrity_encodings(celeb_images)

# Upload user image
user_img = st.file_uploader("Choose your image...", type="jpg")

if user_img is not None:
    st.info("Processing your image...")

    # Convert uploaded image to PIL format
    img = Image.open(user_img)

    # Detect celebrities in the image
    detected_img = detect_celebrities(img, celeb_encodings)

    # Display result
    st.image(detected_img, caption='Detected Celebrity',
             use_container_width=True)

