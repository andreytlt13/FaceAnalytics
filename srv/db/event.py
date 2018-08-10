from sqlalchemy import Column, Integer, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Event(Base):
    __tablename__ = 'events_log'

    id = Column(Integer)
    person_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    log_time = Column(BigInteger, primary_key=True)
    camera_url = Column(String, primary_key=True)

    def __repr__(self):
        return "<Event(id='%d', person_name='%s', age='%d', gender='%s', log_time='%d', camera_url='%s')>" % (
            self.id, self.person_name, self.age, self.gender, self.log_time, self.camera_url)
