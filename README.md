# Real-Time-Face-Recognition
Face Recognition using OpenCV and face_recognition


A real time face recognition system is capable of identifying or verifying a person from a video frame.
To recognize the face in a frame, first you need to detect whether the face is present in the frame. 
If it is present, mark it as a region of interest (ROI), extract the ROI and process it for facial recognition


### Methodology 

I applied simple Machine Learning Algorithms on it. I scrapped the encodings of known images and Train my model on it. It gives 89.67 % accuracy on testing data.
Using face_recognition python library can help in real-time face recognition. 


### Prerequisites 
    Ubuntu:18.04 / Env - Python3 

1. I use "face_recognition" python module. I give some Training Images to it. I use "SVM" algorithm for Training the array data.
-----you can easily install it by "sudo pip3 install face_recognition".


2. I handle the webcam by OpenCV and Process all Image Processing By it.
-----you can easily install its main and contrib modules by "sudo pip3 install opencv-contrib-python".


### Step By Step Guide 

Step 1. install the reqiurements.txt file.
		$ pip3 install -r requirements.txt

Step 2. give the image's path and name in friendsreco.py file on line - 10.


Step 3. Run the python code.
		$ python3 friendsreco.py


################################################################################


Further Enhancemets --- 
			Frozen Training Process
			Resources Based Labelized Training
