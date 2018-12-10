from sqlalchemy import *#create_engine, MetaData, Table, Column
from sqlalchemy.orm import sessionmaker
from common import config_parser

CONFIG = config_parser.parse()
DEFAULT_PATH = 'sqlite:///surveillance.db'
Session = sessionmaker()


class EventDBLogger:

    def __init__(self, engine=DEFAULT_PATH):
        self.engine = create_engine(engine, echo=True)
        self.metadata = MetaData(self.engine, reflect=True)
        self.conn = self.engine.connect()
        self.inspector = inspect(self.engine)

    def create_table(self, name):
        if not self.engine.dialect.has_table(self.engine, name):
            table = Table(name, self.metadata,
                  Column('id', Integer, primary_key=True, autoincrement=True),
                  Column('object_id', Integer),
                  Column('event_time', BigInteger),
                  Column('enter', Integer),
                  Column('exit', Integer),
                  Column('y', BigInteger),
                  Column('x', BigInteger)
                  )
            self.metadata.create_all()
        else:
            table = self.metadata.tables[name]
        return table

    def insert(self, table, dict):

        ins = table.insert().values(
            event_time=dict['event_time'],
            object_id=dict['object_id'],
            enter=dict['enter'],
            exit=dict['exit'],
            y=dict['y'],
            x=dict['x']
        )
        self.conn.execute(ins)
