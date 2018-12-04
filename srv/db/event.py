from sqlalchemy import Column, Integer, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Event(Base):
    __tablename__ = 'events_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    object_id = Column(Integer)
    event_time = Column(BigInteger)
    enter = Column(Integer)
    exit = Column(Integer)

    def __repr__(self):
        return "<Event(id=%d, object_id='%d', event_time='%d', enter='%d', exit='%d')>" % (
            self.id, self.object_id, self.event_time, self.enter, self.exit)


class Event2(Base):
    __tablename__ = 'events_log2'

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer)
    person_name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    log_time = Column(BigInteger)

    def __repr__(self):
        return "<Event(id=%d, person_id='%d', person_name='%s', age='%d', gender='%s', log_time='%d')>" % (
            self.id, self.person_id, self.person_name, self.age, self.gender, self.log_time)