# %%
from yolov5 import YOLOv5

# Set paths and parameters
weights_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/yolov5s.pt'
input_video_path = '/home/gyasis/Downloads/VID_20190813_140505.mp4'
output_video_path = '/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/output_video.mp4'




# %%
import torch
import cv2
from tqdm import tqdm

def process_video(input_video_path, output_video_path, model):
    class_names = model.names if hasattr(model, 'names') else model.model.names
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv5 object detection
        results = model(frame)

        # Draw detection results on the frame
        for *xyxy, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = class_names[int(cls)]
            label = f"{class_name} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # Write the frame to the output video
        out.write(frame)

        # Show the frame in real-time
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Set paths and parameters


# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True).to(device)

# Process the video file
process_video(input_video_path, output_video_path, model)

# %%
