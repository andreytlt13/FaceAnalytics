FROM srv_base

WORKDIR /srv
ADD flask_api /srv/flask_api
ADD frame_processing /srv/frame_processing

ENV PYTHONPATH /
ENTRYPOINT [ "python", "./run_flask.py" ]
WORKDIR /srv/flask_api

EXPOSE 9090
