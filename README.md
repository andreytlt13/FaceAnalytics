# FaceAnalytics with dlib and flask

[dlib](http://dlib.net/) is a machine learning library used to detect and recognize the faces

[flask](http://flask.pocoo.org/) is a micro framework to create web page using python

##Install packages

```
pip install -r requirements.txt
```

## Download models
For detection and recognition you need to download this [models](https://drive.google.com/drive/folders/1hEKVTEp6BrlSG12NF1XvmsjJiMVO89bm?usp=sharing)
and put them into  <b> FaceAnalytics/main/model </b>



#### * Optional
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
