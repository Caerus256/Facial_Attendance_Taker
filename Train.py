import face_recognition
import numpy as np
import cv2
import os
import pickle
from time import time

def capture_face(name):
    """
    Capture a face from the webcam and save it to the 'Photos' directory.
    
    Args:
        name (str): The name of the person whose face is being captured.
    
    Returns:
        bool: True if the face was captured successfully, False otherwise.
    """
    print(f'Capturing face for {name}. Please look at the camera for 5 seconds.')

    # Create the 'Photos' directory if it doesn't exist
    photos_dir = os.path.join(os.path.dirname(__file__), 'Photos')
    os.makedirs(photos_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    start_time = time()
    while True:
        ret, frame = cam.read()
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or time() - start_time > 5:
            break

    cam.release()
    cv2.destroyAllWindows()

    if ret:
        image_path = os.path.join(photos_dir, f'{name}.jpg')
        cv2.imwrite(image_path, frame)
        print(f'Face captured and saved as {image_path}')
        return True
    else:
        print('Failed to capture face. Please try again.')
        return False

def add_face_encoding(name, image_path):
    """
    Add a face encoding to the dataset.
    
    Args:
        name (str): The name of the person whose face is being added.
        image_path (str): The path to the image file containing the face.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_faces.dat')

    # Load existing face encodings
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            face_encodings = pickle.load(f)
    else:
        face_encodings = {}

    # Encode the new face
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]

    # Add the new face encoding to the dataset
    face_encodings[name] = encoding

    # Save the updated dataset
    with open(dataset_path, 'wb') as f:
        pickle.dump(face_encodings, f)

    print(f'Face encoding for {name} added to the dataset.')

def delete_face_encoding(name):
    """
    Delete a face encoding from the dataset.
    
    Args:
        name (str): The name of the person whose face encoding is to be deleted.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset_faces.dat')

    # Load existing face encodings
    if os.path.exists(dataset_path):
        with open(dataset_path, 'rb') as f:
            face_encodings = pickle.load(f)
    else:
        face_encodings = {}

    # Delete the face encoding from the dataset
    if name in face_encodings:
        del face_encodings[name]
        print(f'Face encoding for {name} deleted from the dataset.')

        # Save the updated dataset
        with open(dataset_path, 'wb') as f:
            pickle.dump(face_encodings, f)
    else:
        print(f'Face encoding for {name} not found in the dataset.')

def main():
    while True:
        choice = input('(a) Add a user\n(b) Delete a user\n(c) Exit\n').lower()

        if choice == 'a':
            name = input('Please enter your name: ')
            if capture_face(name):
                image_path = os.path.join(os.path.dirname(__file__), 'Photos', f'{name}.jpg')
                add_face_encoding(name, image_path)

        elif choice == 'b':
            name = input('Please enter the name of the user to delete: ')
            delete_face_encoding(name)

        elif choice == 'c':
            print('Exiting...')
            break

        else:
            print('Invalid choice. Please try again.')

if __name__ == '__main__':
    main()