# Attendance using Facial Recognition System

This is a face recognition system built using Python and the face_recognition library. The system can capture faces from a webcam, store face encodings in a dataset, and recognize faces in real-time video.

## Features

- Capture faces from the webcam and save them to the dataset
- Add or delete face encodings from the dataset
- Recognize faces in real-time video using the stored face encodings

## Requirements

- Python 3.x
- OpenCV
- face_recognition
- NumPy
- Pickle

## Installation

1. Clone the repository:
2. Install the required packages:

## Usage

1. Run the `Train.py` script to add or delete face encodings from the dataset: Follow the prompts to capture faces, add them to the dataset, or delete existing encodings.
2. Run the `Face_recognition.py` script to start the face recognition system:

   The system will open a window displaying the video feed from your webcam. It will detect and recognize faces in real-time, drawing bounding boxes and names around the recognized faces.

Press 'q' to exit the program.

## File Structure

- `Face_recognition.py`: The main script for running the face recognition system.
- `Train.py`: The script for adding or deleting face encodings from the dataset.
- `dataset_faces.dat`: The file containing the face encodings and names of known faces.
- `Photos/`: The directory where captured face images are stored.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.
