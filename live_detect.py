"""
live_detect.py  --  Watch YOLO find the ball in real time.

Opens a window and plays a video with a green box drawn on the ball as it moves.
Press q to quit.

Usage:
    python live_detect.py [path_to_video]

If no path is given it uses jungle_video.mp4 in this folder.
Note: on a CPU without a GPU the playback is slower than real speed, because YOLO
runs on every single frame.
"""

import sys
import cv2
from ultralytics import YOLO

# yolov8s (small). The nano model was faster but mistook the ball for a teddy bear.
model = YOLO("yolov8s.pt")

SPORTS_BALL = 32   # COCO class id for a ball
PERSON = 0

video = sys.argv[1] if len(sys.argv) > 1 else "jungle_video.mp4"
cap = cv2.VideoCapture(video)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, classes=[SPORTS_BALL, PERSON], conf=0.3, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("YOLO - press q to quit", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
