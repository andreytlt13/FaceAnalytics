import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.event import Base, Event
from db.event_db_logger import EventDBLogger

engine = create_engine('sqlite:///test.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class TestDBLogger(unittest.TestCase):

    def setUp(self):
        Base.metadata.create_all(engine)

    def test_log_to_db(self):
        logger = EventDBLogger(db_url='sqlite:///test.db')
        logger.log({
            "person_id": 0,
            "person_name": "Jack",
            "age": 69,
            "gender": "Male",
            "log_time": "1533886427"
        })
        logger.log({
            "person_id": 5,
            "person_name": "Judy",
            "age": 27,
            "gender": "Female",
            "log_time": "1533886429"
        })

        self.assertEqual(
            str(session.query(Event).filter_by(person_name='Jack').first()),
            "<Event(id=1, person_id='0', person_name='Jack', age='69', gender='Male', log_time='1533886427')>"
        )
        self.assertEqual(
            str(session.query(Event).filter_by(gender='Female').first()),
            "<Event(id=2, person_id='5', person_name='Judy', age='27', gender='Female', log_time='1533886429')>"
        )

    def tearDown(self):
        for tbl in reversed(Base.metadata.sorted_tables):
            engine.execute(tbl.delete())
