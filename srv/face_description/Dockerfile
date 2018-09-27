FROM srv_base

WORKDIR /srv
ADD db /srv/db
ADD face_description /srv/face_description

ENV PYTHONPATH /srv
ENTRYPOINT [ "python", "./run_descriptor.py" ]
WORKDIR /srv/face_description