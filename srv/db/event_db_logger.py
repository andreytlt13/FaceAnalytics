from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from srv.db.event import Event, Base

Session = sessionmaker()


class EventDBLogger:

    def __init__(self, db_url='sqlite:///surveillance.db') -> None:
        engine = create_engine(db_url, echo=True)
        Base.metadata.create_all(engine)
        Session.configure(bind=engine)

        self.session = Session()

    def log(self, data):
        # with open(json_log, "r") as read_file:
        #     data = json.load(read_file)
        event = Event(
            person_id=int(data['id']), person_name=data['person_name'], age=int(data['age']),
            gender=data['gender'], log_time=int(data['log_time']),
            camera_url=data['camera_url']
        )
        self.session.add(event)
        self.session.commit()
