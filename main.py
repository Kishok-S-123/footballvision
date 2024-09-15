from ultralytics import YOLO
import supervision as sv
import numpy as np


model = YOLO("trained_models/best.pt")
tracker = sv.ByteTrack()
ellipse_annotator = sv.EllipseAnnotator()
label_annotator = sv.LabelAnnotator()



def callback(frame: np.ndarray, _: int):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id 
        in zip(detections.class_id, detections.tracker_id)
        ]
    
    ellipse_frame = ellipse_annotator.annotate(frame.copy(), detections= detections)

    return label_annotator.annotate(ellipse_frame.copy(), detections = detections, labels = labels)


sv.process_video(source_path="Clips/0bfacc_2.mp4", target_path = "tracked/result.mp4", callback = callback)

