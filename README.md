# Real-Time-Face-Recognition
This project contains Face detection and Face recognition using the famous Open Source Libraries OpenCV and face_recognition.


> **Note:** It is not feasable to build customized CNN model and train it using thousands of Images as it'll take a lot of time depending on your hardware. Always prefer Transfer Learning over doing everything from scratch.
 


#### Face Detection
<p> For Face Detection you need to download <a href = "https://opencv.org/">OpenCV</a>. The one used in the project is OpenCV v2. </p>



#### Face Recognition
<p> For Face Recognition you need to download both <a href = "https://opencv.org/">OpenCV</a> and <a href="https://pypi.org/project/face_recognition/">face_recognition</a> libraries. </p> 

I applied simple Machine Learning Algorithms on it. I scrapped the encodings of known images and Train my model on it. It gives 89.67 % accuracy on testing data.


#### Resources 
                             Ubuntu:18.04 / Env - Python3 


#### Step By Step Guide 

* Install the requirements.txt file in each of the Face Detection and Face Recognition directories.
   * $ pip3 install -r requirements.txt



* Do not forget to provide the arguments while running the detection and recognition scripts. Details of the arguments are given in respective README's.



* Run the python code according to the application.
   * $ python3  <file_name>
