import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from srv.db.event import Base, Event
from srv.db.log_events_to_db import EventDBLogger

engine = create_engine('sqlite:///test.db', echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class TestDBLogger(unittest.TestCase):

    def setUp(self):
        Base.metadata.create_all(engine)

    def test_log_to_db(self):
        logger = EventDBLogger('sqlite:///test.db')
        logger.log('resources/log.json')

        self.assertEqual(
            str(session.query(Event).filter_by(person_name='Jack').first()),
            "<Event(id='0', person_name='Jack', age='69', gender='Male', log_time='1533886427', "
            + "camera_url='rtsp://admin:admin@10.101.106.12:554/ch01/1')>"
        )

    def tearDown(self):
        for tbl in reversed(Base.metadata.sorted_tables):
            engine.execute(tbl.delete())
