import cv2
import sys

# Initializing the face cascade from the xml file
# given
# Cascade is just an xml file containing face
# detection data
cascPath = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# To capture a video, you need to create a VideoCapture object.
# Its argument can be either the device index or the name of a video 
# file. Device index is just the number to specify which camera. 
# Normally one camera will be connected (as in my case). So I simply 
# pass 0 (or -1). You can select the second camera by passing 1 and 
# so on. After that, you can capture frame-by-frame. 
# But at the end, don’t forget to release the capture.

# This line sets the video source to the default webcam, which OpenCV can easily capture.

video_capture = cv2.VideoCapture(0)
# video_capture = cv2.VideoCapture(<file_name>)

# A video file can also be passed here, but you need to have 
# ffmpeg installed for that since OpenCV itself cannot decode
# compressed video. Ffmpeg acts as the front end for OpenCV, 
# and, ideally, it should be compiled directly into OpenCV. 
# This is not easy to do, especially on Windows.

while True:
    # Capture frame-by-frame
    # video_capture.read() returns actual video frame read, in 
    # each loop ( as we are inside a while loop) and a return code
    # which is a bool(True | False). 
    # If frame is read correctly, return value will be True.
    ret, frame = video_capture.read()
    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # 0xFF == ord('q') simply means script will exit if we press
    # q key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the capture as mentioned above.
video_capture.release()
cv2.destroyAllWindows()
