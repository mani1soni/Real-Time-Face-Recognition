#!/usr/bin/python3
import face_recognition
import cv2


# Getting a reference to webcam
video_capture = cv2.VideoCapture(0)

# Loading a sample picture for Trainig
manish_image0 = face_recognition.load_image_file("/home/manish/Documents/projectabs/738.jpg")
manish_image1 = face_recognition.load_image_file("/home/manish/Documents/projectabs/manish1.jpg")
manish_image2 = face_recognition.load_image_file("/home/manish/Documents/projectabs/manish2.jpg")
ankit_image0 = face_recognition.load_image_file("/home/manish/Documents/projectabs/ankit.jpg")
ankit_image1 = face_recognition.load_image_file("/home/manish/Documents/projectabs/ankit1.jpg")
ankit_image2 = face_recognition.load_image_file("/home/manish/Documents/projectabs/ankit2.jpg")
jay_image0 = face_recognition.load_image_file("/home/manish/Documents/projectabs/jayg1.jpg")
jay_image1 = face_recognition.load_image_file("/home/manish/Documents/projectabs/jayg2.jpg")
jay_image2 = face_recognition.load_image_file("/home/manish/Documents/projectabs/jayg3.jpg")
manish_face_encoding0 = face_recognition.face_encodings(manish_image0)[0]
manish_face_encoding1 = face_recognition.face_encodings(manish_image1)[0]
manish_face_encoding2 = face_recognition.face_encodings(manish_image2)[0]
ankit_face_encoding0 = face_recognition.face_encodings(ankit_image0)[0]
ankit_face_encoding1 = face_recognition.face_encodings(ankit_image1)[0]
ankit_face_encoding2 = face_recognition.face_encodings(ankit_image2)[0]
jay_face_encoding0 = face_recognition.face_encodings(jay_image0)[0]
jay_face_encoding1 = face_recognition.face_encodings(jay_image1)[0]
jay_face_encoding2 = face_recognition.face_encodings(jay_image2)[0]

# Creating arrays of known face encodings and their names
known_face_encodings = [
    manish_face_encoding0,
    manish_face_encoding1,
    manish_face_encoding2,
    ankit_face_encoding0,
    ankit_face_encoding1,
    ankit_face_encoding2,
    jay_face_encoding0,
    jay_face_encoding1,
    jay_face_encoding2,
]
known_face_names = [
    "Manish",
    "Manish",
    "Manish",
    "Ankit",
    "Ankit",
    "Ankit",
    "Jaydeep",
    "Jaydeep",
    "Jaydeep"
]

# Initializing some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Taking a single frame of video
    ret, frame = video_capture.read()

    # Resizing frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color 
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Finding all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "unknown face"

            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Displaying the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Drawing a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Drawing a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Displaying the resulting image
    cv2.imshow('Hello', frame)

    # Hit 'q'  to quit!!!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
