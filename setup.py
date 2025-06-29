# setup.py

import os
import joblib
import face_recognition
import cv2
from celeb_images import images_list as celeb_images


class CelebrityEncoder:
    """
    A class to encode celebrity images into face encodings for facial recognition.

    Attributes
    ----------
    celeb_images : List[str]
        A list of file paths to celebrity images.
    encodings : List[Tuple[str, np.ndarray]]
        A list of tuples where each contains a celebrity name and their face encoding.
    """

    def __init__(self, celeb_images):
        """
        Initialize the CelebrityEncoder with a list of celebrity image paths.

        Parameters
        ----------
        celeb_images : List[str]
            A list of file paths to celebrity images.
        """
        self.celeb_images = celeb_images
        self.encodings = []

    def load_and_encode(self):
        """
        Load each image, resize it, and encode the face.

        This method processes all images in `celeb_images`, detects faces,
        and stores their encodings along with the corresponding celebrity names.
        """
        for image_path in self.celeb_images:
            try:
                # Load the image using face_recognition
                celeb_image = face_recognition.load_image_file(image_path)
                # Resize the image for faster processing
                celeb_image_resized = cv2.resize(
                    celeb_image, (0, 0), fx=0.5, fy=0.5
                )
                # Encode the face using face_recognition
                celeb_encoding = face_recognition.face_encodings(celeb_image_resized)[
                    0]
            except IndexError:
                print(f"No face found in {image_path}. Skipping this image.")
                continue
            # Extract the celebrity name from the filename
            name = os.path.splitext(os.path.basename(image_path))[0]
            # Store the encoding along with the name
            self.encodings.append((name, celeb_encoding))

    def save_encodings(self, output_path="./data/celebrity_encodings.joblib"):
        """
        Save the celebrity face encodings to a file.

        Parameters
        ----------
        output_path : str, optional
            The path where the encodings will be saved. Defaults to "data/celebrity_encodings.joblib".
        """
        joblib.dump(self.encodings, output_path)
        print(f"Encodings saved to {output_path}")


def run():
    """
    Entry point for the script.

    Loads celebrity images, encodes them into face encodings,
    and saves the result to a file.
    """
    encoder = CelebrityEncoder(celeb_images)
    encoder.load_and_encode()
    encoder.save_encodings()


if __name__ == "__main__":
    run()
