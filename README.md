# Football Juggle Detection

Count how many times a player juggles a football in a video, using a pretrained
YOLO model to find the ball and simple math to count the juggles. No training needed.

## Demo

Simple version (counts every juggle from the ball height):

![simple demo](assets/simple_demo.gif)

Fusion version (tracks the feet too, so it detects when the ball leaves the feet
and resets the streak, for example on a failed trick):

![fusion demo](assets/fusion_demo.gif)

## How it works

The ball is already a known class in YOLO (`sports ball`, class 32 in COCO), so no
training is required. The pipeline is:

1. Run YOLO on every frame to get the ball position.
2. Track the ball height over time. Juggling makes a wave, one bump per juggle.
3. Take the derivative of the height. The top of a juggle is where the velocity
   goes from positive to negative (a downward zero crossing).
4. Keep only the bumps that stand out (prominence filter), which removes small
   wiggles like the ball being picked up or settling on the floor.

The **fusion** version adds a pose model to track the feet. A juggle is then a
low point where the ball is close to a foot. A break is a low point where the ball
is far from the feet (it dropped or rolled away), which resets the streak counter.

## Notebooks

- `juggle_counter.ipynb` — the simple version (height only). Fast.
- `juggle_counter_fusion.ipynb` — the advanced version (feet + ball). About two
  times slower because it runs two models per frame, but it detects breaks.
- `live_detect.py` — play a video with the ball box drawn in real time.

## Setup

Python 3.12 is required (see `requirements.txt` for why).

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put a **sharp** video named `jungle_video.mp4` in this folder and run a notebook.
A sharp video matters: on a blurry clip the ball turns into a white blob in the air
and YOLO loses it, so detection drops a lot. Good light or a fast shutter helps.

## Notes

- Everything runs on the CPU, so processing a clip takes a few minutes. A GPU would
  be much faster, but it is not required.
- The ball detection works best on a clear, well lit video with the ball in frame.
