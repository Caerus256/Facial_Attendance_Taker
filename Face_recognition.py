import face_recognition
import cv2
import os
import numpy as np
import pickle

def load_known_faces():
    """
    Load known face encodings and names from the dataset.

    """
    dataset_path = os.path.dirname(__file__) + '/dataset_faces.dat'
    with open(dataset_path, 'rb') as data:
        all_face_encodings = pickle.load(data)
    known_face_encodings = np.array(list(all_face_encodings.values()))
    known_face_names = list(all_face_encodings.keys())
    return known_face_encodings, known_face_names

def recognize_faces(frame, known_face_encodings, known_face_names):
    """
    Recognize faces in the given frame using the known face encodings and names.

    """
    # Resize frame for faster processing
    rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        # Check if the face matches any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Use the name of the known face with the smallest distance
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"

        face_names.append(name)

    return face_locations, face_names

def draw_boxes_and_names(frame, face_locations, face_names):
    """
    Draw bounding boxes and names on the frame.

    """
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 10, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

def main():
    """
    Run face recognition on live video from the webcam.
    
    """
    # Load known face encodings and names
    known_face_encodings, known_face_names = load_known_faces()

    # Open the webcam
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if process_this_frame:
            # Recognize faces in the frame
            face_locations, face_names = recognize_faces(frame, known_face_encodings, known_face_names)

            # Draw bounding boxes and names on the frame
            draw_boxes_and_names(frame, face_locations, face_names)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Process every other frame to save time
        process_this_frame = not process_this_frame

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()