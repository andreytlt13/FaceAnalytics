FROM python:3.6

WORKDIR /srv

ADD common /srv/common
ADD config /srv/config
ADD flask_api /srv/flask_api
ADD frame_processing /srv/frame_processing
ADD requirements.txt /srv

ENV PYTHONPATH /srv
ENTRYPOINT [ "python", "./run_flask.py" ]

RUN apt-get update \
        && apt-get install -y --fix-missing \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        python3-dev \
        python3-numpy \
        -y libgtk2.0-dev \
        && pip3 install --trusted-host pypi.python.org -r requirements.txt \
        && apt-get clean && rm -rf /tmp/* /var/tmp/*


WORKDIR /
ENV OPENCV_VERSION="3.4.1"
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
        && unzip ${OPENCV_VERSION}.zip \
        && mkdir -p /opencv-${OPENCV_VERSION}/cmake_binary \
        && cd /opencv-${OPENCV_VERSION}/cmake_binary \
        && cmake -DBUILD_TIFF=ON \
          -DBUILD_opencv_java=OFF \
          -DWITH_CUDA=OFF \
          -DENABLE_AVX=ON \
          -DWITH_OPENGL=ON \
          -DWITH_OPENCL=ON \
          -DWITH_IPP=ON \
          -DWITH_TBB=ON \
          -DWITH_EIGEN=ON \
          -DWITH_V4L=ON \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DCMAKE_BUILD_TYPE=RELEASE \
          -DCMAKE_INSTALL_PREFIX=$(python3.6 -c "import sys; print(sys.prefix)") \
          -DPYTHON_EXECUTABLE=$(which python3.6) \
          -DPYTHON_INCLUDE_DIR=$(python3.6 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
          -DPYTHON_PACKAGES_PATH=$(python3.6 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. \
        && make install \
        && rm /${OPENCV_VERSION}.zip \
        && rm -r /opencv-${OPENCV_VERSION}

WORKDIR /srv/flask_api

EXPOSE 9090
