# FaceAnalytics with dlib and flask
[dlib](http://dlib.net/) is a machine learning library used to detect and recognize the faces

[flask](http://flask.pocoo.org/) is a micro framework to create web page using python

##Install packages Used
* dlib
```
conda install -c conda-forge dlib
```


* Flask
```
conda install Flask
```

* Numpy
```
pip install numpy
```

* OpenCV
```
conda install -c menpo opencv -y
pip install opencv-python
```

* Scipy
```
pip install scipy
```

##Download models
For detection and recognition you need to download this [models](https://drive.google.com/drive/folders/1PO1zneiefNjcNdf9PZz-2Y2a7ns8umxe?usp=sharing)
```
 pip install face_recognition_models
```

<b> Person-reid </b>

For person-reid task you need to download weights of pretrained [models](https://drive.google.com/drive/folders/1F09uLEfc_QeAUHkw22BDFsPWjsLyrG6Q?usp=sharing) 
and put them into <b> tmp_exp/person_reid/model  </b>

## Instructions
```
FaceAnalytics$ export FLASK_APP=srv/flask_api/run_flask.py
FaceAnalytics$ flask run
```
##For webcam streaming
```
http://0.0.0.0:9090/video_stream?0
```

#### *
If you don't have a support CUDA on your mashine, install dlib without support cuda

Clone the code from github:
```
git clone https://github.com/davisking/dlib.git
```
Build the main dlib library
```
cd dlib
mkdir build; cd build; cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```
Build and install the Python extensions:
```
cd ..
python3 setup.py install --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA
```
