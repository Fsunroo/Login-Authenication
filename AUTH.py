import face_recognition
import cv2
import numpy as np
import os

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
mohammad_image = face_recognition.load_image_file("mohammad.jpg")
mohammad_face_encoding = face_recognition.face_encodings(mohammad_image)[0]


known_face_encodings = [
    mohammad_face_encoding
]
known_face_names = [
    "Mohammd"
]
stat=False
i=0
while i<9:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            stat=True
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        


    # Display the resulting image
    cv2.imshow('authentication', frame)
    if stat==True:
        # time.sleep(2)
        exit()
        # break

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i+=1

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
os.system("shutdown /r /t 1")
