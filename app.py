# app.py

import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import os

# Import the images list
from celeb_images import images_list as celeb_images


def load_celebrity_encodings(celeb_images):
    celebrity_encodings = []

    for image_path in celeb_images:
        celeb_image = face_recognition.load_image_file(image_path)
        try:
            celeb_encoding = face_recognition.face_encodings(celeb_image)[0]
        except IndexError:
            print(f"No face found in {image_path}. Skipping this image.")
            continue
        name = os.path.splitext(os.path.basename(image_path))[
            0
        ]  # Remove file extension
        celebrity_encodings.append((name, celeb_encoding))

    return celebrity_encodings


def detect_celebrities(img, celeb_encodings):
    # Convert PIL image to RGB
    img_rgb = np.array(img)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(img_rgb)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        # See if the face is a match for the known faces
        matches = []
        name = "Unknown"

        for celeb_name, celeb_encoding in celeb_encodings:
            match = face_recognition.compare_faces([celeb_encoding], face_encoding)
            matches.extend(match)

        # Use the first match if the best (most recent) one is not from just a single pixel
        if True in matches:
            first_match_index = matches.index(True)
            name = celeb_encodings[first_match_index][0].replace(
                "_", " "
            )  # Replace "_" with space

        # Calculate the text size
        (text_width, text_height), _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_DUPLEX, 1.0, 1
        )

        # Calculate the centered position
        text_x = left + (right - left) // 2 - text_width // 2
        text_y = bottom + 35

        # Draw a box around the face
        cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(
            img_rgb,
            (text_x - 6, text_y + 10),
            (text_x + text_width + 6, text_y + text_height + 45),
            (0, 0, 255),
            cv2.FILLED,
        )  # Adjust rectangle position to be below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(
            img_rgb, name, (text_x + 6, text_y + 35), font, 1.0, (255, 255, 255), 1
        )

    # Convert back to PIL image
    img_pil = Image.fromarray(img_rgb)

    return img_pil


st.title("Celebrity Face Recognition")
st.write("Upload an image to detect any known celebrities!")

# Upload user's image
user_img = st.file_uploader("Choose your image...", type=["jpg", "jpeg", "png"])

if user_img is not None:
    celeb_encodings = load_celebrity_encodings(celeb_images)

    img = Image.open(user_img)
    detected_img = detect_celebrities(img, celeb_encodings)

    st.image(detected_img, caption="Detected Celebrity", use_container_width=True)
