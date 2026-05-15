import datetime
import time
import wave

import uvicorn
from eventemitter import EventEmitter
from fastapi import FastAPI, WebSocket

app = FastAPI()
sound_event_emitter = EventEmitter()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last = time.time()
    time_avg = 0.0
    counter = 10
    channels = 1

    bits = 16
    sample_rate = 44100

    exit_loop = False
    num_samples = 0
    save_every_n_seconds = 30

    while not exit_loop:
        with wave.open(
            f"bruhica/output_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')}.wav",
            "wb",
        ) as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(bits // 8)
            wf.setframerate(sample_rate)

            while True:
                try:
                    now = time.time()
                    time_avg += now - last
                    last = now

                    data = await websocket.receive_bytes()

                    wf.writeframes(data)
                    num_samples += len(data)
                    print(
                        num_samples,
                        sample_rate * save_every_n_seconds * channels * (bits // 8),
                        num_samples
                        / (sample_rate * save_every_n_seconds * channels * (bits // 8))
                        * 100,
                    )

                    sound_event_emitter.emit("stream_bytes", data)

                    if num_samples >= sample_rate * save_every_n_seconds * channels * (
                        bits // 8
                    ):
                        print(f"Saved {save_every_n_seconds} seconds of audio")
                        num_samples = 0
                        break

                except Exception as e:
                    print(f"Connection closed: {e}")
                    exit_loop = True
                    break
        counter -= 1
        if counter == 0:
            print(f"Avg time per message: {int(time_avg/10*1000)}ms")
            counter = 10
            time_avg = 0.0


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
