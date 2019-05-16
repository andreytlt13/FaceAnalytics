import pandas as pd
from sqlalchemy import *
from common import config_parser
import json

CONFIG = config_parser.parse()
DEFAULT_PATH = CONFIG['sqlite_db']


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
                  Column('x', BigInteger),
                  Column('names', String),
                  Column('name', String),
                  Column('stars', String),
                  Column('description', String)
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
            x=dict['x'],
            names=str(dict['names']),
            name=dict['name'],
            stars=dict['stars'],
            description=dict['description']
        )
        self.conn.execute(ins)

    def select(self, table, start_date, end_date):
        select_st = table.select().where(and_(
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

