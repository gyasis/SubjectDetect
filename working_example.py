# %%
from yolov5 import YOLOv5

# Set paths and parameters
weights_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/yolov5s.pt'
input_video_path = '/home/gyasis/Downloads/VID_20190322_193105.mp4'
output_video_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/output_video.mp4'




# %%
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import torch
import pandas as pd
from tqdm import tqdm

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
    # Initialize video capture and get video properties
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize DeepSORT
    # max_age: This parameter represents the maximum number of consecutive frames for which a track remains unconfirmed before it is deleted. A higher value makes the tracker more tolerant to temporary occlusions, but can also increase the chances of ID switches. You can try adjusting this value based on the nature of your video and occlusion scenarios.

    
    # You have to play with these values depneding on the best results
    
    tracker = DeepSort(
    max_age = 10,
    n_init=3,
    nms_max_overlap=0.3,
    max_cosine_distance=0.2,
    nn_budget=None
)

    # Initialize DataFrame to store detections
    detections = []

    frame_id = 0
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Process detections
        bbs = []
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            bbs.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())))

        # Update tracks
        tracks = tracker.update_tracks(bbs, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(int, ltrb)

            # Append detection to DataFrame
            detections.append({
                'frame_id': frame_id,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'track_id': track_id,
            })

            # Draw bounding box and track ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write frame with annotations
        out.write(frame)

        # Display the frame in real-time
        cv2.imshow('frame', frame)
        cv2.waitKey(1)  # Add a small delay to show the frame

        frame_id += 1

    # Release video resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Create DataFrame from detections list
    detections_df = pd.DataFrame(detections)

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

def merge_overlapping_clips(clips_df, overlap_threshold):
    merged_clips = []
    sorted_clips = clips_df.sort_values('in').reset_index(drop=True)

    current_clip = sorted_clips.iloc[0]
    for _, next_clip in sorted_clips.iloc[1:].iterrows():
        if next_clip['in'] - current_clip['out'] <= overlap_threshold:
            current_clip['out'] = max(current_clip['out'], next_clip['out'])
        else:
            merged_clips.append(current_clip)
            current_clip = next_clip

    merged_clips.append(current_clip)
    return pd.DataFrame(merged_clips)


def markers(df, fps, area_threshold, middle_percent, duration_threshold, overlap_threshold):
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
    duration_df = middle_df.groupby('track_id').filter(lambda x: len(x) >= duration_threshold * fps)

    # Calculate in and out timecodes
    result = duration_df.groupby('track_id').agg({
        'frame_id': [lambda x: x.min() / fps, lambda x: x.max() / fps]
    }).reset_index()
    result.columns = ['id', 'in', 'out']

    # Merge overlapping clips
    result = merge_overlapping_clips(result, overlap_threshold)

    return result
# %%
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

detections_df['frame_width'] = frame_width
detections_df['frame_height'] = frame_height


fps = 30  # Set your video's FPS here
area_threshold = 10  # Set your area threshold here
middle_percent = 1  # Set your middle percent threshold here
duration_threshold = 2  # Set your minimum duration threshold here (in seconds)

result_df = markers(detections_df, fps, area_threshold, middle_percent, duration_threshold, overlap_threshold=0.01)

  # %%
result_df.to_csv('result.csv', index=False)
# %%
import cv2

import cv2

def play_clips(video_path, result_df, fps):
    cap = cv2.VideoCapture(video_path)

    # Sort the DataFrame by 'id' column
    sorted_result_df = result_df.sort_values(by='id').reset_index(drop=True)
    
    print(sorted_result_df)

    for _, row in sorted_result_df.iterrows():
        object_id = row['id']
        start_time = row['in']
        end_time = row['out']

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print("I'm taking a break")
                break

            # Add the object ID label to the frame
            label = f"ID: {object_id}"
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Clip', frame)

            # Press 'q' to stop playback
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

            current_frame += 1

       
    cap.release()
    cv2.destroyAllWindows()


# %%
play_clips(input_video_path, result_df, fps)
# %%
