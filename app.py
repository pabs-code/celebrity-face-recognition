import logging
import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image
import joblib

# Prevent Streamlit from rerunning on every input change
st.set_page_config(page_title="Celebrity Face Recognition", layout="wide")

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize session state for model loading
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# Initialize session state for progress bar
if "progress" not in st.session_state:
    st.session_state.progress = 0.0
if "progress_message" not in st.session_state:
    st.session_state.progress_message = ""

# Progress bar and status update function


def update_progress(message, value):
    st.session_state.progress = value
    st.session_state.progress_message = message

# Singleton class for celebrity recognition with model loading and progress feedback


class SingletonCelebrityRecognizer:
    _instance = None

    def __new__(cls, encoding_file="./data/celebrity_encodings.joblib"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.encoding_file = encoding_file
            cls._instance.celeb_encodings = None  # Will be loaded lazily
            cls._instance.loaded = False  # Flag to indicate if model is loaded
        return cls._instance

    def load_model(self):
        """
        Load the precomputed celebrity encodings from a file.

        Returns:
            list: List of (celebrity_name, face_encoding) tuples.
        """
        try:
            # Show progress bar and message
            update_progress("Loading celebrity encodings...", 0.2)
            self.celeb_encodings = joblib.load(self.encoding_file)
            logging.info("Celebrity encodings loaded successfully.")
            update_progress("Model loading complete!", 1.0)
        except Exception as e:
            logging.error(f"Failed to load celebrity encodings: {e}")
            st.error(
                "Error loading precomputed celebrity encodings. Please check the file path.")
            raise  # Re-raise to let the calling code handle it

    def detect_faces(self, image):
        """
        Detect and recognize faces in the input image.

        Args:
            image (PIL.Image): The uploaded image to analyze.

        Returns:
            PIL.Image: Modified image with celebrity labels.
        """
        if not self.loaded:
            try:
                self.load_model()
                self.loaded = True
            except Exception as e:
                logging.error(f"Error loading model during detection: {e}")
                st.error(
                    "Failed to load celebrity encodings. Cannot detect faces.")
                return image

        try:
            img_rgb = np.array(image)
            face_locations = face_recognition.face_locations(img_rgb)
            face_encodings = face_recognition.face_encodings(
                img_rgb,
                face_locations
            )

            # Get the image height to calculate font scale
            img_height = img_rgb.shape[0]  # Get the image height

            # Use Streamlit slider to control font size
            font_scale = max(0.5, min(1, (img_height / 255)))

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = "Unknown"
                matches = []

                for celeb_name, celeb_encoding in self.celeb_encodings:
                    match = face_recognition.compare_faces(
                        [celeb_encoding],
                        face_encoding
                    )
                    matches.extend(match)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.celeb_encodings[first_match_index][0].replace(
                        "_", " ").replace("-", " ")

                # Draw the box and label
                (text_width, text_height), _ = cv2.getTextSize(
                    name,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_scale,
                    thickness=1
                )

                text_x = left + (right - left) // 2 - text_width // 2
                text_y = bottom + 35

                cv2.rectangle(img_rgb, (left, top),
                              (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(
                    img_rgb,
                    (text_x - 6, text_y + 10),
                    (text_x + text_width + 6, text_y + text_height + 45),
                    (0, 0, 255),
                    cv2.FILLED
                )
                font = cv2.FONT_HERSHEY_DUPLEX
                # Slightly increase font thickness for better visibility
                cv_a = max(1, int(font_scale * 1.2))
                cv2.putText(img_rgb, name, (text_x + 6, text_y + 35),
                            font, font_scale, (255, 255, 255), cv_a)

            img_pil = Image.fromarray(img_rgb)
            return img_pil
        except Exception as e:
            logging.error(f"Error during face detection: {e}")
            st.error("An error occurred while detecting faces.")
            return image


def main():
    """
    Main function to run the Streamlit application.

    This function sets up the UI, handles user input,
    and displays results using the CelebrityRecognizer class.
    """
    st.title("Celebrity Face Recognition")

    # Description of the app
    st.markdown("""
        ### 🎬 Celebrity Face Recognition App

        Upload one or more images to detect and identify any known celebrities in the photos.  
        The app uses pre-trained celebrity face encodings to match faces with known identities.

        **What you can expect:**
        - The app will load your image(s) and detect faces.
        - If a face matches any known celebrity, it will be labeled with their name.
        - Unknown faces are labeled as "Unknown".
    """)

    # Initialize the singleton celebrity recognizer
    recognizer = SingletonCelebrityRecognizer()

    # Allow multiple image uploads
    uploaded_images = st.file_uploader("Choose your images...",
                                       type=["jpg", "jpeg", "png"],
                                       accept_multiple_files=True)

    if uploaded_images:
        st.markdown("### 📸 Uploaded Images")
        for i, image in enumerate(uploaded_images):
            try:
                original_image = Image.open(image)
                st.markdown(f"### Image {i + 1}")
                col1, col2 = st.columns(2)

                with col1:
                    resized_original = original_image.resize((500, 500))
                    st.image(
                        resized_original,
                        caption=f"Original Image {i + 1}",
                        use_container_width=True
                    )

                with st.spinner(f"Processing image {i + 1}..."):
                    detected_img = recognizer.detect_faces(original_image)
                    resized_detected = detected_img.resize((500, 500))

                with col2:
                    st.image(
                        resized_detected,
                        caption=f"Detected Image {i + 1}",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error processing image {i + 1}: {e}")

    # Display progress message if available
    if st.session_state.progress_message:
        with st.status(st.session_state.progress_message, state="complete"):
            pass

    # Display progress bar
    if st.session_state.progress > 0:
        st.progress(st.session_state.progress)


if __name__ == "__main__":
    main()
