from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common import config_parser
from db.event import Event, Base

CONFIG = config_parser.parse()
DEFAULT_PATH = 'sqlite:///surveillance.db' #CONFIG['../db/db.sqlite3']
Session = sessionmaker()


class EventDBLogger:

    def __init__(self, db_url=DEFAULT_PATH) -> None:
        engine = create_engine(db_url, echo=True)
        Base.metadata.create_all(engine)
        Session.configure(bind=engine)

        self.session = Session()

    def log(self, data):
        event = Event(
            person_id=int(data['person_id']), person_name=data['person_name'], age=int(data['age']),
            gender=data['gender'], log_time=int(data['log_time'])
        )
        self.session.add(event)
        self.session.commit()
