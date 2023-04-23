import argparse
import cv2
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
import time

def load_yolo_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(frame, model, threshold=0.5):
    transform = transforms.Compose([transforms.ToTensor()])
    frame_tensor = transform(frame)
    output = model([frame_tensor])

    detections = []

    for i in range(len(output[0]['boxes'])):
        if output[0]['scores'][i] > threshold:
            detections.append({
                'box': output[0]['boxes'][i].tolist(),
                'label': output[0]['labels'][i].item(),
                'score': output[0]['scores'][i].item()
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center = (frame_width / 2, frame_height / 2)

    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        detections = detect_objects(frame, model)
        all_detections.extend(detections)

        for det in detections:
            cv2.rectangle(frame, (int(det['box'][0]), int(det['box'][1])), (int(det['box'][2]), int(det['box'][3])),
                          (255, 0, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # cv2.destroyAllWindows()

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

