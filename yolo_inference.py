from ultralytics import YOLO

model = YOLO("YOLOv8x")

results = model.predict("Bundesliga/Clips/0a2d9b_9.mp4", save=True)

print(results[0])

for box in results[0].boxes

# git test change

