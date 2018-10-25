import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from srv.db.db_report_reader import DBReportReader
from srv.db.event import Base, Event

TEST_DB = 'sqlite:///test.db'

engine = create_engine(TEST_DB, echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class TestDBReportReader(unittest.TestCase):
    def setUp(self):
        Base.metadata.create_all(engine)
        session.add(Event(person_id=0, person_name='Jack', age=69, gender='Male', log_time=1533886427))
        session.add(Event(person_id=1, person_name='John', age=35, gender='Male', log_time=1533886429))
        session.add(Event(person_id=1, person_name='John', age=35, gender='Male', log_time=1533886435))
        session.add(Event(person_id=3, person_name='Jill', age=25, gender='Female', log_time=1533886435))
        session.add(Event(person_id=3, person_name='Jill', age=25, gender='Female', log_time=1533886439))
        session.commit()

        self.db_report_reader = DBReportReader(db_url=TEST_DB)

    def test_read_faces_count_by_date(self):
        expected = [(1533886429, 1), (1533886435, 2)]

        result = self.db_report_reader.read_faces_count_by_date(1533886428, 1533886437)

        self._assertEquals(result, expected)

    def test_read_genders_by_date(self):
        expected = [(1533886429, 'Male'), (1533886435, 'Male'), (1533886435, 'Female')]

        result = self.db_report_reader.read_genders_by_date(1533886428, 1533886437)

        self._assertEquals(result, expected)

    def test_read_ages_by_date(self):
        expected = [(1533886429, 35), (1533886435, 35), (1533886435, 25)]

        result = self.db_report_reader.read_ages_by_date(1533886428, 1533886437)

        self._assertEquals(result, expected)

    def test_read_person_by_date(self):
        expected = [(1533886429, 'John'), (1533886435, 'John'), (1533886435, 'Jill')]

        result = self.db_report_reader.read_person_by_date(1533886428, 1533886437)

        self._assertEquals(result, expected)

    def test_read_all_by_date(self):
        expected = [
            (1, 'John', 35, 'Male', 1533886429), (1, 'John', 35, 'Male', 1533886435),
            (3, 'Jill', 25, 'Female', 1533886435)
        ]

        result = self.db_report_reader.read_all_by_date(1533886428, 1533886437)

        self._assertEquals(result, expected)

    def _assertEquals(self, result, expected):
        i = 0
        for r in result:
            self.assertEqual(r, expected[i])
            i += 1

    def tearDown(self):
        for tbl in reversed(Base.metadata.sorted_tables):
            engine.execute(tbl.delete())
