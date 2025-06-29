# Celebrity Face Recognition App

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit Version](https://img.shields.io/badge/streamlit-1.20.0%2B-green.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

## Table of Contents

- [Celebrity Face Recognition App](#celebrity-face-recognition-app)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
    - [What is it?](#what-is-it)
    - [How It Works:](#how-it-works)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Folder Structure](#folder-structure)
  - [Installation](#installation)
  - [Running Script](#running-script)
  - [Expectations When Running This App](#expectations-when-running-this-app)
  - [Demo](#demo)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)
  - [Notes](#notes)
    - [Limitations of Using `face_recognition`](#limitations-of-using-face_recognition)
    - [Customization](#customization)
  - [Troubleshooting](#troubleshooting)

## About the Project

This application leverages face recognition technology to identify celebrities from uploaded images.  It's built using Python, Streamlit, and the `face_recognition` library uses pre-trained models for face encoding.

### What is it?

The Celebrity Face Recognition App is a Streamlit application that allows users to upload images and automatically detect and identify faces, matching them against a pre-trained database of celebrity face encodings. The application clearly displays the recognized faces with their names, or labels them as "Unknown" if a match is not found.

### How It Works:

1.  **Image Upload:** The user uploads one or more image files (JPG, JPEG, PNG) through the Streamlit UI.
2.  **Face Detection:** The application utilizes the `face_recognition` library to detect faces within the uploaded images.
3.  **Face Encoding:** For each detected face, the `face_recognition` library generates a unique face encoding, a numerical representation of the face.
4.  **Comparison:**  The generated face encodings are compared against a pre-trained database of celebrity face encodings loaded from a `joblib` file.
5.  **Identification:** If a matching face encoding is found, the application identifies the face and displays the name of the celebrity. If no match is found, the face is labeled as "Unknown."
6. **Display Results**: The app displays both the original and the processed images with the celebrity's names labeled on the faces.

## Features

*   **Multiple Image Upload:** Supports uploading multiple images simultaneously.
*   **Celebrity Identification:** Automatically identifies celebrities from uploaded images.
*   **Unknown Face Detection:**  Labels unidentified faces as "Unknown."
*   **User-Friendly Interface:**  Intuitive Streamlit UI for easy image uploading and viewing results.
*   **Responsive Layout:** Optimized for various screen sizes.
*   **Clear Results Display**: The app displays both the original and processed images so you can see what the AI detected.
*   **Modular Design**: Separation of model training, face detection logic, and UI components.
*  **Progressive Feedback** during processing with progress bars.

## Getting Started

### Prerequisites

*   **Python 3.8+:**  Make sure you have Python 3.8 or a later version installed.  You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
*   **pip:**  Python's package installer (usually included with Python installations).
*   **Git:**  For cloning the repository.

Before running the application, ensure you have installed the following libraries:

```bash
pip install streamlit face_recognition opencv-python numpy scikit-learn pillow
```

### Folder Structure

The project structure is as follows:

```
celebrity_face_recognition/
│
├── app.py                   # Streamlit App for face detection UI
├── setup.py                 # Script to encode and save celebrity images as model
├── celeb_images.py          # List of image paths used to train the model
├── data/                    # Folder where trained model is saved
│   └── celebrity_encodings.joblib  # Pre-trained face encodings
├── img/                     # Folder containing celebrity images (can be updated by users)
│   ├── Tom_Cruise.jpeg
│   └── ... (other celebrity images)
```

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd celebrity_face_recognition
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Script

To run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser.

## Expectations When Running This App

*   **Recognition Accuracy:** The accuracy of celebrity recognition depends on the quality of the uploaded images and the similarity of the detected faces to the pre-trained database.
*   **Performance:**  Processing multiple high-resolution images may take some time.
*   **Pre-trained Model:** The app uses a pre-trained model for celebrity recognition.  The list of celebrities available for recognition is limited to those included in the `celebrity_encodings.joblib` file.
*   **Data File**:  The initial `celebrity_encodings.joblib` file may contain a smaller number of celebrities.  It may require training a new file to contain more celebrities.

## Demo


<img width="971" alt="Screenshot 2025-06-28 at 5 29 53 PM" src="https://github.com/user-attachments/assets/751bfbc8-0669-4ebf-abd7-043c1454a4ee" />

<img width="985" alt="Screenshot 2025-06-28 at 5 30 02 PM" src="https://github.com/user-attachments/assets/b35675ed-d3cc-49ca-83c9-60b492a7ffa9" />

## Acknowledgments

- [**face_recognition library**](https://github.com/ageitgey/face-recognition) – Used for face encoding and recognition.
- [**Streamlit**](https://streamlit.io) – Used to build the interactive web app.
- [**OpenCV (cv2)**](https://opencv.org) – Used for image processing.
- [**scikit-learn (joblib)**](https://scikit-learn.org) – Used for model serialization.


## License

This project is licensed under the [MIT License](LICENSE).

## Notes


### Limitations of Using `face_recognition`

- The `face_recognition` library is built on **dlib**, which may have performance and memory issues for large datasets.
- It uses a pre-trained model that is not fine-tuned on specific use cases (e.g., celebrity recognition).
- It may have **accuracy issues** with low-quality or partially visible faces.

### Customization

- This model is trained using only a few images listed in `celeb_images.py`. If you want to add more celebrities or update the model:
  - Add their image files to the `img/` directory.
  - Update `celeb_images.py` with the new image paths.
  - Run the training script again using:

```bash
python setup.py
```

This will re-encode and save the new model in `data/celebrity_encodings.joblib`.

---

## Troubleshooting

- **Error: No face found in image.**  
  Ensure your images have clear, frontal faces and are of high quality.

- **Model not loading.**  
  Verify that the file `data/celebrity_encodings.joblib` exists and is not corrupted.

- **Face detection is slow.**  
  Consider reducing image resolution or using GPU acceleration (if supported).

---
