# %%
from yolov5 import YOLOv5
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import pandas as pd
from tqdm import tqdm

# Set paths and parameters
weights_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/yolov5s.pt'
input_video_path = '/home/gyasis/Downloads/VID_20190813_140505.mp4'
output_video_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/output_video.mp4'


def get_video_info(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    resolution = None
    if width == 1280 and height == 720:
        resolution = '720p'
    elif width == 1920 and height == 1080:
        resolution = '1080p'
    elif width >= 3840 and height >= 2160:
        resolution = '4k'
    else:
        resolution = f"{width}x{height}"

    video_info = {
        'resolution': resolution,
        'width': width,
        'height': height,
        'fps': fps,
    }

    return video_info

def print_video_info(video_info):
    formatted_info = f"Video information:\nResolution: {video_info['resolution']}\nWidth: {video_info['width']} px\nHeight: {video_info['height']} px\nFPS: {video_info['fps']}"
    print(formatted_info)

def process_video(input_video_path, output_video_path, model):
    
    class_names = model.names if hasattr(model, 'names') else model.model.names
    
    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    detections_list = []
    tracker = DeepSort(
        max_age = 30,
        n_init=3,
        nms_max_overlap=0.3,
        max_cosine_distance=0.2,
        nn_budget=None
    )

    for frame_id in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = video.read()
        if not ret:
            break

        # Perform object detection
        model.conf = 0.25
        model.iou = 0.45
        detections = model(frame)

        # Process detections
        bboxes = []
        for *xyxy, conf, cls in detections.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            width = x2 - x1
            height = y2 - y1
            class_name = class_names[int(cls)]
            bboxes.append(([x1, y1, width, height], conf.item(), int(cls)))

        tracks = tracker.update_tracks(bboxes, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            track_id = track.track_id
            class_id = track.get_class()
            class_name = class_names[class_id]
            x1, y1, x2, y2 = map(int, ltrb)

            detections_list.append({
                'frame_id': frame_id,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'confidence': track.get_confidence(),
                'class': class_id,
                'name': class_name,
                'id': track_id
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {track_id}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        out.write(frame)

    video.release()
    out.release()

    detections_df = pd.DataFrame(detections_list)
    return detections_df


# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True).to(device)

# Get and print video information
video_info = get_video_info(input_video_path)
print_video_info(video_info)

# Process the video file
detections_df = process_video(input_video_path, output_video_path, model)

# Do something with the detections_df (e.g., save it to a file)
detections_df.to_csv('detections.csv', index=False)


# %%
import pandas as pd

def markers(df, fps, area_threshold, middle_percent, duration_threshold):
    # Calculate the middle of the frame
    middle_area_min = (1 - middle_percent) / 2
    middle_area_max = 1 - middle_area_min

    # Filter based on area threshold
    df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
    area_df = df[df['area'] >= area_threshold]

    # Filter based on middle_percent
    middle_df = area_df[(area_df['xmin'] >= area_df['frame_width'] * middle_area_min) &
                        (area_df['xmax'] <= area_df['frame_width'] * middle_area_max) &
                        (area_df['ymin'] >= area_df['frame_height'] * middle_area_min) &
                        (area_df['ymax'] <= area_df['frame_height'] * middle_area_max)]

    # Group the DataFrame by object ID and filter based on the minimum duration threshold
    duration_df = middle_df.groupby('name').filter(lambda x: len(x) >= duration_threshold * fps)

    # Calculate in and out timecodes
    result = duration_df.groupby('name').agg({
        'frame_id': [lambda x: x.min() / fps, lambda x: x.max() / fps]
    }).reset_index()

    # Rename columns
    result.columns = ['name', 'in_timecode', 'out_timecode']

    return result
