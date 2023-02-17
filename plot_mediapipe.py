import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList
import time
import argparse
import os
import glob
from random import sample
from tqdm import tqdm
from multiprocessing import Pool
import json
from pathlib import Path
import numpy as np

def draw_mediapipe_landmarks(video_details):
  mediapipe_filepath = video_details['mediapipe_filepath']
  video_filepath = video_details['video_filepath']
  new_video_filepath = video_details['new_video_filepath']
  show_overlay = video_details.get('show_overlay', False)
  no_source_video = video_details.get('no_src', True)
  print(new_video_filepath)

  mp_drawing = mp.solutions.drawing_utils
  mp_holistic = mp.solutions.holistic

  # For video input:
  holistic = mp_holistic.Holistic(
      min_detection_confidence=0.5, min_tracking_confidence=0.1)
  if not no_source_video:
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
  else:
    fps = 30
  result = cv2.VideoWriter(filename=new_video_filepath, 
                         fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
                         fps=fps, frameSize=(480, 640))
  start = time.time()
  num_frames = 0
  pose_null = 0

  generate_mediapipe = not os.path.isfile(mediapipe_filepath)
  print(mediapipe_filepath)
  if not generate_mediapipe:
    with open(mediapipe_filepath) as f:
      json_data = json.load(f)
      # print("COUNT", sum(1 for _ in json_data), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  else:
    json_data = {}

  if no_source_video:
    for frame_num in json_data:
      image = np.zeros((1944, 2592, 3), np.uint8)
      # Convert the image to OpenCV RGB.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      pose = json_data[frame_num]["pose"]
      if pose != {}:
        pose_landmarks = NormalizedLandmarkList(
          landmark = [NormalizedLandmark(x=pose[str(x)][0], y=pose[str(x)][1], z=pose[str(x)][2]) for x in range(33)]
        )
        mp_drawing.draw_landmarks(
          image, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
      left_hand = json_data[frame_num]["landmarks"]["0"]
      if left_hand != {}:
        left_hand_landmarks = NormalizedLandmarkList(
          landmark = [NormalizedLandmark(x=left_hand[str(x)][0], y=left_hand[str(x)][1], z=left_hand[str(x)][2]) for x in range(21)]
        )
        mp_drawing.draw_landmarks(
          image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      right_hand = json_data[frame_num]["landmarks"]["1"]
      if right_hand != {}:
        right_hand_landmarks = NormalizedLandmarkList(
          landmark = [NormalizedLandmark(x=right_hand[str(x)][0], y=right_hand[str(x)][1], z=right_hand[str(x)][2]) for x in range(21)]
        )
        mp_drawing.draw_landmarks(
          image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      image = cv2.resize(image, (480, 640))
      result.write(image)
      if show_overlay:
        cv2.imshow('Overlay', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
      num_frames += 1
  else:
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break

      # Convert the BGR image to RGB.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      generate_mediapipe = not os.path.isfile(mediapipe_filepath)

      if generate_mediapipe:
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results_mp = holistic.process(image)
      image.flags.writeable = True
      # Draw landmark annotation on the image.
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if generate_mediapipe:
        pose_landmarks = results_mp.pose_landmarks
        left_hand_landmarks = results_mp.left_hand_landmarks
        right_hand_landmarks = results_mp.right_hand_landmarks
      else:
        frame_data = json_data[str(num_frames)]
        pose_landmarks = NormalizedLandmarkList(
          landmark = [NormalizedLandmark(x=frame_data["pose"][str(x)][0], y=frame_data["pose"][str(x)][1], z=frame_data["pose"][str(x)][2]) for x in range(33)]
        )
        left_hand_json = frame_data["landmarks"]["0"]
        if left_hand_json == {}:
          left_hand_landmarks = None
        else:
          left_hand_landmarks = NormalizedLandmarkList(
            landmark = [NormalizedLandmark(x=left_hand_json[str(x)][0], y=left_hand_json[str(x)][1], z=left_hand_json[str(x)][2]) for x in range(21)]
          )
        right_hand_json = frame_data["landmarks"]["1"]
        if right_hand_json == {}:
          right_hand_landmarks = None
        else:
          right_hand_landmarks = NormalizedLandmarkList(
            landmark = [NormalizedLandmark(x=right_hand_json[str(x)][0], y=right_hand_json[str(x)][1], z=right_hand_json[str(x)][2]) for x in range(21)]
          )
      mp_drawing.draw_landmarks(
          image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      mp_drawing.draw_landmarks(
          image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
      mp_drawing.draw_landmarks(
          image, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
      if pose_landmarks is None:
        pose_null += 1

      #define the screen resulation
      screen_res = 1280, 720
      scale_width = screen_res[0] / image.shape[1]
      scale_height = screen_res[1] / image.shape[0]
      scale = min(scale_width, scale_height)
      #resized window width and height
      window_width = int(image.shape[1] * scale)
      window_height = int(image.shape[0] * scale)
      #cv2.WINDOW_NORMAL makes the output window resizealbe
      window_width = int(cap.get(3))
      window_height = int(cap.get(4))
      image = cv2.flip(image, 1)
      result.write(image)
      # if show_overlay:
      #   cv2.imshow('Overlay', image)
      # if cv2.waitKey(5) & 0xFF == 27:
      #   break
      num_frames += 1
    cap.release()
  
  end = time.time() - start
  print("Number of times pose is none = " + str(pose_null))
  print("Time taken = " + str(end))
  print("Total frames = " + str(num_frames))
  print("Frames processed per second = " + str(num_frames/end))
  holistic.close()
  result.release()

  return new_video_filepath


if __name__ == "__main__":
  """ This is executed when run from the command line """
  parser = argparse.ArgumentParser()

  parser.add_argument("-s", "--video_src", default="/data/sign_language_videos/split")

  parser.add_argument("-m", "--mediapipe_src", default="/data/sign_language_videos/mediapipe")

  parser.add_argument("-d", "--dest", default="/data/sign_language_videos/parquet_overlay")

  parser.add_argument("-n", "--no_src_videos", action='store_true')

  parser.add_argument("-p", "--partial_videos", action='store_true')

  args = parser.parse_args()
  print(args)

  user_dirs = [os.path.join(args.mediapipe_src, name) for name in os.listdir(args.mediapipe_src)
            if os.path.isdir(os.path.join(args.mediapipe_src, name))]

  for user_dir in user_dirs:
    dest_dir = os.path.join(args.dest, user_dir.split('/')[-1].split('.')[0])
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if not args.no_src_videos:
      src_files = glob.glob(os.path.join(args.video_src, "**", "*.mp4"))

      # change if subset
      if args.partial_videos:
        processed_src_files = sample(src_files, int(0.1*len(src_files)))
      else:
        processed_src_files = src_files

      dest_files = []
      for file in tqdm(processed_src_files):
        dest_files.append(os.path.join(dest_dir, file.split('/')[-1]))

      mapped = []
      sign_count = {}
      for src_f, dest_f in zip(processed_src_files, dest_files):
        curr_user, curr_sign, _ = str(src_f).split('/')[-1].split('-')
        if curr_sign not in sign_count:
          sign_count[curr_sign] = 1
        curr_recording_count = sign_count[curr_sign]
        sign_count[curr_sign] += 1
        mediapipe_filepath = os.path.join(user_dir, f"{curr_user}-singlesign", curr_sign, f"{curr_user}.{curr_sign}.singlesign.{str(curr_recording_count).zfill(8)}.data")
        mapped.append({'video_filepath': str(src_f), 'mediapipe_filepath': str(mediapipe_filepath), 'new_video_filepath': str(dest_f), 'no_src': True})
    else:
      mp_files = glob.glob(os.path.join(user_dir, '**', "*.data"))

      if args.partial_videos:
        processed_mp_files = sample(mp_files, int(0.1*len(mp_files)))
      else:
        processed_mp_files = mp_files

      dest_files = []
      for file in tqdm(processed_mp_files):
        d_f = os.path.join(dest_dir, '.'.join(file.split('/')[-1].split('.')[:-1])+'.mp4')
        print(d_f)
        dest_files.append(d_f)

      mapped = []
      for mp_f, dest_f in zip(processed_mp_files, dest_files):
        mapped.append({'video_filepath': '/', 'mediapipe_filepath': str(mp_f), 'new_video_filepath': str(dest_f), 'no_src': True})

    for mediapipe_info in tqdm(mapped):
      draw_mediapipe_landmarks(mediapipe_info)
