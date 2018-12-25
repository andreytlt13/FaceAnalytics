import asyncio
import pickle

import websockets


async def get_full_report():
    async with websockets.connect(
            'ws://localhost:8181/full_report') as websocket:
        start_dt_timestamp = str(input("Start date timestamp for report: "))
        end_dt_timestamp = str(input("End date timestamp for report: "))

        await websocket.send(start_dt_timestamp)
        await websocket.send(end_dt_timestamp)

        result = await websocket.recv()
        print(pickle.loads(result))


asyncio.get_event_loop().run_until_complete(get_full_report())
