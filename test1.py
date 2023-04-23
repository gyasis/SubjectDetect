import argparse
import cv2
import torch

from torchvision import transforms
from collections import defaultdict
import time
from yolov5 import YOLOv5
from tqdm import tqdm

import os
import subprocess

def download_yolo_weights(weights_path):
    if not os.path.exists(weights_path):
        print("Downloading YOLOv5 weights...")
        subprocess.run(['wget', '-O', weights_path, 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt'])
        print("Weights downloaded.")
    else:
        print("YOLOv5 weights already available.")

weights_path = 'yolov5s.pt'
download_yolo_weights(weights_path)

def load_yolo_model():
    model = YOLOv5(weights_path, device="cuda" if torch.cuda.is_available() else "cpu")
    return model



def detect_objects(frame, model):
    results = model.predict(frame)
    detections = []

    for *box, conf, cls in results.xyxy[0].tolist():
        detections.append({
            'box': box,
            'class': int(cls),
            'conf': conf
        })

    return detections

def rank_subjects(detections, frame_center, duration_weight=0.5, position_weight=0.5):
    subjects = defaultdict(lambda: {'duration': 0, 'position_score_sum': 0})

    for det in detections:
        box_center = ((det['box'][0] + det['box'][2]) / 2, (det['box'][1] + det['box'][3]) / 2)
        position_score = 1 - (abs(frame_center[0] - box_center[0]) + abs(frame_center[1] - box_center[1])) / (
                frame_center[0] + frame_center[1])

        subjects[det['label']]['duration'] += 1
        subjects[det['label']]['position_score_sum'] += position_score

    for label, subject in subjects.items():
        subject['score'] = (subject['duration'] * duration_weight) + (
                subject['position_score_sum'] * position_weight)

    ranked_subjects = sorted(subjects.items(), key=lambda x: x[1]['score'], reverse=True)

    return ranked_subjects

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file '{video_path}'. Please check the file path and format.")
        return []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center = (frame_width / 2, frame_height / 2)

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

    all_detections = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="Processing video frames"):
        ret, frame = cap.read()

        if not ret:
            break

        detections = detect_objects(frame, model)
        all_detections.extend(detections)

        for det in detections:
            cv2.rectangle(frame, (int(det['box'][0]), int(det['box'][1])), (int(det['box'][2]), int(det['box'][3])),
                          (255, 0, 0), 2)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

        try:
            cv2.imshow("Real-time output", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(f"Error displaying real-time output: {e}")
            print("Continuing without real-time display...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    ranked_subjects = rank_subjects(all_detections, frame_center)
    return ranked_subjects



def main():
    parser = argparse.ArgumentParser(description='Object Detection in Videos using YOLO')
    parser.add_argument('video_path', help='Path to the video file')
    args = parser.parse_args()

    model = load_yolo_model()
    video_path = args.video_path

    start_time = time.time()
    ranked_subjects = analyze_video(video_path, model)
    end_time = time.time()

    print(f"Time taken to analyze the video: {end_time - start_time:.2f} seconds")
    print("Ranked Subjects:")
    for rank, (label, subject) in enumerate(ranked_subjects, start=1):
        print(f"Rank: {rank}, Label: {label}, Score: {subject['score']}, Duration: {subject['duration']}, Position Score Sum: {subject['position_score_sum']}")

if __name__ == '__main__':
    main()
