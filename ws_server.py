import asyncio
import json
import websockets
import pandas as pd
from prepare_data import get_scores
import os

clients = set()

async def handler(ws):
    print("Client connected")
    clients.add(ws)
    try:
        await ws.wait_closed()
    finally:
        clients.remove(ws)
        print("Client disconnected")

async def producer():
    live_path = "processed_datasets/live.csv"
    last_len = 0

    while True:
        if os.path.exists(live_path):
            # Skip if file is empty
            if os.path.getsize(live_path) == 0:
                await asyncio.sleep(0.5)
                continue

            try:
                df = pd.read_csv(live_path)
            except pd.errors.EmptyDataError:
                # File exists but has no data yet
                await asyncio.sleep(0.5)
                continue

            if len(df) > last_len:
                new_row = df.iloc[-1:]
                scored = get_scores(new_row)

                score_value = float(scored["anomaly_score"].iloc[0])

                msg = {"score": score_value}
                print("Sending:", msg)

                if clients:
                    # Send to all connected clients
                    await asyncio.gather(*[c.send(json.dumps(msg)) for c in clients])

                last_len = len(df)

        await asyncio.sleep(0.5)  # avoid busy-waiting

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket running at ws://localhost:8765")
    await producer()

asyncio.run(main())