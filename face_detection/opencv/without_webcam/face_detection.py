# Installing OpenCV is the hardest part as 
# there are several compatibility issues, 
# i.e 32/64 bit differences, other libraries 
# could be of older versions, etc.
import cv2
import sys

# Passing image and cascade names as command-line arguments
# imagePath should be dir location of the image you want detection for.
imagePath = sys.argv[1]

# For face detection keep:
cascPath = '../haarcascade_frontalface_default.xml'

# Initializing the face cascade from the xml file
# given via the arguments in cascPath.
# Cascade is just an xml file containing face
# detection data
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image. Many operations in OpenCV 
# are done in grayscale.
# Remember OpenCV loads images as BGR images.
image = cv2.imread(imagePath)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# IMPORTANT PART

# Detect faces in the image
# The detectMultiScale function is a general function that detects objects. Since we are calling it on the face cascade, that’s what it detects.
# scale factor compensates for different different sizes of face in the image.
# The detection algorithm takes up a small window os size == minSize and hovers it throughout the image. But generally minSize is smaller than the size of the face, hence we define a parameter called minNeighbors which says how many objects are detected near the current one before it declares the face found. 

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

# The function returns a list of rectangles in which it believes it found a face.
print("Found {0} faces!".format(len(faces)))

# Next, we will loop over where it thinks it found something and
# draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces Detected", image)
cv2.waitKey(0)
