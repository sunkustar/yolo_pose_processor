import asyncio
import cv2
import base64
import websockets
import json
import time

# ------------------- Config
MANAGER_WS_URL = "ws://127.0.0.1:8005/ws_stream"
VIDEO_PATH = "/home/sunku/vertex_ai_server/test2i.mp4"
FRAME_INTERVAL = 0  # seconds

async def stream_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    async with websockets.connect(MANAGER_WS_URL, max_size=2**24) as ws:
        frame_idx = 0
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Encode frame to JPEG + base64
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                print(f"Frame {frame_idx} → Failed to encode")
                continue
            frame_bytes = buffer.tobytes()

            await ws.send(frame_bytes)  # send raw JPEG bytes

            processing_time_ms = (time.time() - start_time) * 1000
            print(f"Frame {frame_idx} sent in {processing_time_ms:.2f} ms")

            frame_idx += 1
            await asyncio.sleep(FRAME_INTERVAL)

        cap.release()
        print(f"✅ Finished streaming {frame_idx} frames")

if __name__ == "__main__":
    asyncio.run(stream_video())
