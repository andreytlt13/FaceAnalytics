import pandas as pd
from sqlalchemy import *

from common import config_parser

CONFIG = config_parser.parse()
DEFAULT_PATH = 'sqlite:///data/db/4_floor.db'  # surveillance.db'


class EventDBLogger:

    def __init__(self, db_name):
        engine = 'sqlite:///data/db/{}.db'.format(db_name)
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
                          Column('centroid_x', BigInteger),
                          Column('centroid_y', BigInteger),
                          )
            self.metadata.create_all()
        else:
            table = self.metadata.tables[name]
        return table

    def create_table_event_logger(self):
        name = 'event_logger'
        if not self.engine.dialect.has_table(self.engine, name):
            table = Table(name, self.metadata,
                          Column('id', Integer, primary_key=True, autoincrement=True),
                          Column('object_id', Integer),
                          Column('event_time', BigInteger),
                          Column('centroid_x', BigInteger),
                          Column('centroid_y', BigInteger),
                  )
            self.metadata.create_all()
        else:
            table = self.metadata.tables[name]
        return table

    def create_table_recognized_logger(self):
        name = 'recognized_logger'
        if not self.engine.dialect.has_table(self.engine, name):
            table = Table(name, self.metadata,
                          Column('id', Integer, ForeignKey("event_logger.object_id"), autoincrement=True),
                          Column('name', Text),
                          Column('description', Text),
                          Column('stars', Text),
                          Column('img_path', Text),
                          )
            self.metadata.create_all()
        else:
            table = self.metadata.tables[name]
        return table

    def insert_describe(self, table, dict):
        ins = table.insert().values(
            id=dict['event_time'],
            name=dict['object_id'],
            description=dict['centroid_y'],
            stars=dict['centroid_x'],
            img_path=dict['centroid_x'])
        print(ins)
        self.conn.execute(ins)

    def insert_event(self, table, dict):
        ins = table.insert().values(
            event_time=dict['event_time'],
            object_id=dict['object_id'],
            centroid_y=dict['centroid_y'],
            centroid_x=dict['centroid_x'])
        print(ins)
        self.conn.execute(ins)

    def insert_describe(self, table, dict):
        ins = table.insert().values(
            event_time=dict['event_time'],
            object_id=dict['object_id'],
            centroid_y=dict['centroid_y'],
            centroid_x=dict['centroid_x'])
        print(ins)
        self.conn.execute(ins)

    def select(self, table, start_date, end_date):
        select_st = table.select().where(
            and_(
            table.c.event_time > start_date,
            table.c.event_time < end_date))
        res = self.conn.execute(select_st).fetchall()
        if len(res) == 0:
            return pd.DataFrame().to_json(orient='records')
        else:
            df = pd.DataFrame(res)
            df.columns = res[0].keys()
            j = df.to_json(orient='records')
            return j