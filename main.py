from ultralytics import YOLO
import supervision as sv
import numpy as np
import torch
import cv2

model = YOLO("trained_models/best.pt")
tracker = sv.ByteTrack()
ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator()



# def callback(frame: np.ndarray, _: int):
#     results = model(frame)[0]
#     detections = sv.Detections.from_ultralytics(results)
#     detections = tracker.update_with_detections(detections)

#     labels = [
#         f"{tracker_id} {results.names[class_id]}"
#         for class_id, tracker_id 
#         in zip(detections.class_id, detections.tracker_id)
#         ]
    
#     ellipse_frame = ellipse_annotator.annotate(frame.copy(), detections= detections)

#     return label_annotator.annotate(ellipse_frame.copy(), detections = detections, labels = labels)

## collecting detection boxes:
"""
1. Collect bounding box information on all detections of player
2. perform kmeans clustering of subsection of image corresponding with box coordinates
 
"""

## 1.


# cap = cv2.VideoCapture('Clips/0bfacc_2.mp4')
# while cap.isOpened():
#     ret, frame = cap.read()

#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
    
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imwrite("assets/muhframe.jpg", frame)
#     cap.release()
#     break




########

playerBoxes = [torch.Tensor.cpu(box.xyxy) for box in model("assets/muhframe.jpg")[0].boxes if box.cls == torch.tensor([2.], device="cuda:0")]
print(f"player detections: {playerBoxes}")

sample_frame = cv2.imread("assets/muhframe.jpg")
sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)

print(sample_frame.shape)
print(np.shape(playerBoxes[0]))
print(playerBoxes[0])
# print(f"playerBoxes[0]: {playerBoxes[0][0][0]}")
print(f"x1, y1: {playerBoxes[0][0][0]}, {playerBoxes[0][0][1]}")
print(f"x2, y2: {playerBoxes[0][0][2]}, {playerBoxes[0][0][3]}")

sample_player = sample_frame[int(playerBoxes[0][0][1]):int(playerBoxes[0][0][3]) , int(playerBoxes[0][0][0]):int(playerBoxes[0][0][2]) ,:]

cv2.imwrite("assets/sample_player_img.jpg", sample_player)





## 2.



# sv.process_video(source_path="Clips/0bfacc_2.mp4", target_path = "tracked/result.mp4", callback = callback)

