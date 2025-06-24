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

## About the Project

This application leverages face recognition technology to identify celebrities from uploaded images.  It's built using Python, Streamlit, and the `face_recognition` library.

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

## Getting Started

### Prerequisites

*   **Python 3.8+:**  Make sure you have Python 3.8 or a later version installed.  You can download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
*   **pip:**  Python's package installer (usually included with Python installations).
*   **Git:**  For cloning the repository.

### Folder Structure

The project structure is as follows:

```
celebrity_face_recognition/
├── data/
│   └── celebrity_encodings.joblib  # Pre-trained celebrity face encodings file
├── .gitignore
├── README.md
├── main.py  # Main application script
├── requirements.txt # List of required packages.
└── setup.py # encode celebrity images into face encodings for face recogniton.
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
streamlit run main.py
```

This will open the application in your default web browser.

## Expectations When Running This App

*   **Recognition Accuracy:** The accuracy of celebrity recognition depends on the quality of the uploaded images and the similarity of the detected faces to the pre-trained database.
*   **Performance:**  Processing multiple high-resolution images may take some time.
*   **Pre-trained Model:** The app uses a pre-trained model for celebrity recognition.  The list of celebrities available for recognition is limited to those included in the `celebrity_encodings.joblib` file.
*   **Data File**:  The initial `celebrity_encodings.joblib` file may contain a smaller number of celebrities.  It may require training a new file to contain more celebrities.

## Demo

A live demo of the application will be hosted on Streamlit Cloud soon.

## Acknowledgments

*   **Face Recognition Library:**  We greatly appreciate the developers of the `face_recognition` library for providing a powerful and easy-to-use tool for face detection and recognition.  [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)
*   **Streamlit Community:** We thank the Streamlit community for their invaluable contributions to the development of this fantastic framework. [https://streamlit.io/](https://streamlit.io/)

## License

This project is licensed under the [MIT License](LICENSE).

