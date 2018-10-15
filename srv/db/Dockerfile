FROM srv_base

WORKDIR /srv
ADD db /srv/db

ENV PYTHONPATH /srv

ENTRYPOINT [ "python", "./db_api.py" ]
WORKDIR /srv/db

EXPOSE 8181
