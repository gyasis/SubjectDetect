# %%
from yolov5 import detect

results = detect(source='/home/gyasis/Downloads/VID_20190813_140505.mp4', weights='/media/gyasis/Blade 15 SSD/Users/gyasi/Google Drive (not syncing)/Collection/SubjectDetect/yolov5s.pt', conf=0.25)

for result in results:
    print(result)
# %%
