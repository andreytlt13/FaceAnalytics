#docker build --cache-from faceanalytics:latest --tag=faceanalytics:$CI_BUILD_REF_NAME --tag faceanalytics:latest .
FROM inutano/wget

RUN apk add --no-cache unzip

RUN mkdir /data && cd /data && wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1o7WyUESApAWtP9Cb2oEDugvd08xLqQuN' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1o7WyUESApAWtP9Cb2oEDugvd08xLqQuN" -O model.zip && rm -rf /tmp/cookies.txt

RUN mkdir /unzip && unzip /data/model.zip -d /unzip 


FROM python:3.6

COPY main /FaceAnalytics/main
COPY rest_api /FaceAnalytics/rest_api
COPY requirements.txt /FaceAnalytics/
COPY --from=0 /unzip/model /FaceAnalytics/main/model
COPY Dockerfile /


RUN apt-get update && \
    apt-get install -y cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN cd /FaceAnalytics && \
    pip3 install -r requirements.txt

WORKDIR /FaceAnalytics/main

CMD [ "python3", "./run.py" ]
