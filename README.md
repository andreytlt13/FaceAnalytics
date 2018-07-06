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
* Scikit-image
```
conda install scikit-image
```

##Download models
You need download this [models](https://github.com/davisking/dlib-models)
```
 pip install face_recognition_models
```

## Instructions
```
FaceAnalytics$ export FLASK_APP=srv/webapp/run.py
FaceAnalytics$ flask run
 
```


####*
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