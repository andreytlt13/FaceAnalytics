import asyncio
import pickle

import websockets

from common import config_parser
from db.db_report_reader import DBReportReader

CONFIG = config_parser.parse()
db_report_reader = DBReportReader(db_url=CONFIG['db_url'])


async def full_report(web_socket, _):
    start_dt_timestamp = int(await web_socket.recv())
    end_dt_timestamp = int(await web_socket.recv())
    result = db_report_reader.read_all_by_date(start_dt_timestamp, end_dt_timestamp)
    await web_socket.send(pickle.dumps(result))


start_server = websockets.serve(full_report, host='0.0.0.0', port=8181)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
