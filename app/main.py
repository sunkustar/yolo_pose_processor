import asyncio
import torch
import numpy as np
import json
import time
import websockets
from ultralytics import YOLO
from tracker import PersonTracker
import cv2
import functools
import traceback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR_WS_PORT = 8001

print(f"✅ Processor running on device: {DEVICE}")

# Load YOLO model and tracker once
model = YOLO("yolov8n-pose.pt")
tracker = PersonTracker(max_history=5)

# ------------------- Helper functions for classification
def get_angle(p1, p2, p3):
    """Calculate the angle between three points (p2 is the vertex)."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return np.nan
    return np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0)) * 180 / np.pi

def is_sitting(kp, torso_height):
    """
    Determine if a person is sitting.
    A person is likely sitting if their knees are bent significantly and
    their hips are lower than or close to their knees' vertical position.
    """
    # Keypoints: 11-LHip, 12-RHip, 13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle
    l_hip, r_hip = kp[11], kp[12]
    l_knee, r_knee = kp[13], kp[14]
    l_ankle, r_ankle = kp[15], kp[16]

    # Check knee angles
    left_knee_angle = get_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = get_angle(r_hip, r_knee, r_ankle)

    # If knees are bent significantly (e.g., < 130 degrees), it's a strong sign of sitting.
    # We check if at least one is valid and bent.
    is_knee_bent = (not np.isnan(left_knee_angle) and left_knee_angle < 130) or \
                   (not np.isnan(right_knee_angle) and right_knee_angle < 130)

    # Check vertical position of hips relative to knees
    avg_hip_y = (l_hip[1] + r_hip[1]) / 2
    avg_knee_y = (l_knee[1] + r_knee[1]) / 2
    hip_knee_dy = (avg_knee_y - avg_hip_y) / torso_height

    # If hips are above or only slightly below knees (normalized), it suggests sitting.
    # A standing person would have a large positive hip_knee_dy.
    return is_knee_bent and hip_knee_dy < 0.5

# ------------------- Scale-invariant action classifier
def classify_actions_batch(keypoints_list):
    """
    Classifies actions for a batch of keypoints using more robust geometric heuristics.
    """
    actions = []
    for kp in keypoints_list:
        # Keypoints: 5-LShoulder, 6-RShoulder, 11-LHip, 12-RHip, 13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle
        required_indices = [5, 6, 11, 12, 13, 14, 15, 16]
        if kp is None or len(kp) < 17 or any(np.isnan(kp[idx]).any() for idx in required_indices):
            actions.append("unknown")
            continue

        try:
            # Use vertical distance between shoulders and hips for a stable torso height
            torso_height = abs(((kp[5][1] + kp[6][1]) / 2) - ((kp[11][1] + kp[12][1]) / 2))
            torso_height = max(torso_height, 1e-3)

            if is_sitting(kp, torso_height):
                actions.append("sitting")
            else:
                # Differentiate between walking and standing
                # Check vertical distance between knees. If significant, person is likely walking.
                knee_dy = abs(kp[13][1] - kp[14][1]) / torso_height

                if knee_dy > 0.15: # Threshold for knee separation indicating a step
                    actions.append("walking")
                else:
                    actions.append("standing")
        except Exception:
            # Log the error for debugging but still return a safe default
            traceback.print_exc()
            actions.append("unknown")
    return actions

# ------------------- Frame Processing (CPU/GPU-Bound)
def process_frame(frame_bytes: bytes):
    """
    Decodes, runs inference, and tracks persons in a single frame.
    This is a blocking, CPU/GPU-intensive function.
    """
    processing_start_time = time.time()

    # 1. Decode the JPEG image from bytes
    frame_np = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame_np is None:
        return None

    # 2. YOLO inference (CPU/GPU-bound)
    results = model.predict(frame_np, device=DEVICE, verbose=False)
    
    boxes, keypoints_list = [], []
    if results and results[0].boxes:
        for res, kp in zip(results[0].boxes, results[0].keypoints.xy):
            if float(res.conf) < 0.5:
                continue
            boxes.append(list(map(int, res.xyxy[0])))
            keypoints_list.append(kp.cpu().numpy())

    # 3. Classify actions on the current frame's keypoints
    actions = classify_actions_batch(keypoints_list)

    # 4. Update tracker to get smoothed actions and consistent IDs
    tracked_persons = tracker.match_and_update(boxes, keypoints_list, actions)

    processing_time_ms = (time.time() - processing_start_time) * 1000
    return {"persons": tracked_persons, "processing_time_ms": processing_time_ms}


# ------------------- Processor WS (Async I/O)
async def processor_ws():
    """
    Sets up the WebSocket server and handles connections.
    """
    loop = asyncio.get_running_loop()

    async def handle_connection(websocket):
        print(f"[PROCESSOR] 🤝 New connection from {websocket.remote_address}")
        frame_counter = 0
        async for frame_bytes in websocket:
            handler_start_time = time.time()

            # Run the blocking `process_frame` function in a thread pool
            payload = await loop.run_in_executor(
                None, functools.partial(process_frame, frame_bytes)
            )

            if payload:
                inference_time_ms = payload.pop("processing_time_ms", 0)
                await websocket.send(json.dumps(payload))

            handler_time_ms = (time.time() - handler_start_time) * 1000
            print(f"[PROCESSOR] Frame {frame_counter}: "
                  f"Inference: {inference_time_ms:.2f} ms, Total Handler: {handler_time_ms:.2f} ms")
            frame_counter += 1
        print(f"[PROCESSOR] 👋 Connection closed from {websocket.remote_address}")

    server = await websockets.serve(handle_connection, "0.0.0.0", PROCESSOR_WS_PORT, max_size=2**24)
    print(f"✅ Processor WS listening on port {PROCESSOR_WS_PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(processor_ws())
    except KeyboardInterrupt:
        print("\n[PROCESSOR] Shutting down.")
