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
You need download this [models](https://drive.google.com/drive/folders/1PO1zneiefNjcNdf9PZz-2Y2a7ns8umxe?usp=sharing)
```
 pip install face_recognition_models
```

## Instructions
```
FaceAnalytics$ export FLASK_APP=srv/flask_api/run_flask.py
FaceAnalytics$ flask run
 
```

## For webcam streaming
```
http://0.0.0.0:9090/video_stream?0
```

### Stream and test modes
You can chose 2 modes into config file:
- webcam_mode: 
<b>stream</b> is for web streaming from camera, 
<b>test</b> is for testing app with video file
- detection_mode: 
<b>face</b> uses caffe model for only face detection, 
<b>person</b> uses MobileNetSSD model for person detection
```
webcam_mode : stream #test
detection_mode : face #person
```
####

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

1 - create_camera (class)
      in:
      -url
            --check exist table in db
      state:
      -read_stream (def with while true)
          -object detection (coffe model mobile ssd)
                -tracking objects (def)
                    -filtered (def)
                    -face_recognize (def)
                        --insert event to db (def)
                    
           
          
      
                
                
                
2 - flask api 
            