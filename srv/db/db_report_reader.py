import sys

from sqlalchemy import create_engine, func, and_
from sqlalchemy.orm import sessionmaker

from srv.common import config_parser
from srv.db.event import Base, Event

CONFIG = config_parser.parse()
Session = sessionmaker()


class DBReportReader:

    def __init__(self, db_url='sqlite:////srv/surveillance.db') -> None:
        engine = create_engine(db_url, echo=True)
        Base.metadata.create_all(engine)
        Session.configure(bind=engine)

        self.session = Session()

    def read_faces_count_by_date(self, start_dt_timestamp=0, end_dt_timestamp=sys.maxsize):
        return self.session.query(Event.log_time, func.count(Event.person_id)) \
            .filter(and_(Event.log_time >= start_dt_timestamp, Event.log_time <= end_dt_timestamp)) \
            .group_by(Event.log_time) \
            .all()

    def read_genders_by_date(self, start_dt_timestamp=0, end_dt_timestamp=sys.maxsize):
        return self.session.query(Event.log_time, Event.gender) \
            .filter(and_(Event.log_time >= start_dt_timestamp, Event.log_time <= end_dt_timestamp)) \
            .all()

    def read_ages_by_date(self, start_dt_timestamp=0, end_dt_timestamp=sys.maxsize):
        return self.session.query(Event.log_time, Event.age) \
            .filter(and_(Event.log_time >= start_dt_timestamp, Event.log_time <= end_dt_timestamp)) \
            .all()

    def read_person_by_date(self, start_dt_timestamp=0, end_dt_timestamp=sys.maxsize):
        return self.session.query(Event.log_time, Event.person_name) \
            .filter(and_(Event.log_time >= start_dt_timestamp, Event.log_time <= end_dt_timestamp)) \
            .all()

    def read_all_by_date(self, start_dt_timestamp=0, end_dt_timestamp=sys.maxsize):
        return self.session.query(Event.person_id, Event.person_name, Event.age, Event.gender, Event.log_time) \
            .filter(and_(Event.log_time >= start_dt_timestamp, Event.log_time <= end_dt_timestamp)) \
            .all()
