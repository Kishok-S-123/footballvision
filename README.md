# footballvision
yolov8 vision based football analysis bot using Bundesliga Videos dataset from Kaggle competition: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout

## Project Goals:
1. YOLO object detection- detection of ball and people only involved in the game i.e. players, referee, goalie, etc
2. Player tracking- keeping track of each player (assign an id for each player and keeping track of location)
3. Player colour assignment- detection of team based on shirt colour
4. ball interpolation- interpolating position of ball in frames where it is not picked up by yolo
5. camera movement estimator (optical flow)- keeping track of the movement of the camera in order to calculate distances travelled etc
6. speed and distance estimator- estimating speed of each player as well as total distance travelled


## 1. YOLO Object detection:
- The YOLOv8x pretrained model out of the box is already quite good at detecting people in the video, but it naturally comes with limitations for this application
    - Detection certainty is relatively low, ranging from 40%-70%, which is not ideal
    - The model detects people (among other things) that are not involved in the game such as spectators. We don't want to detect these people as we do not care about their positions or actions during the game
    - The model struggles to detect the football for most of the video, and does not differentiate between players and referees

- To address these concerns, the yolo model can be finetuned on more specific data (roboflow football players detection dataset)
- Due to resource limitatations, training had to be done using google colab notebook with one of their free gpu runtimes
- After training the existing yolov8 model on the more specific football dataset, it is apparent that there are significant increases in 

## Improving annotations
- currently the annotations are very clunky and due to the proximity of the players to each other, the players and ball are occluded.
- This reduces the efficacy of our solution for football analysis, so we would like to change these to something more simple
- this is going to be done by 

## Player tracking
- Tracking can be completed using ByteTrack, which can be utilised effortlessly using the roboflow library

## Player Team Detection:
- Next, in order to be able to produce more meaningful inferences/analytics, it would be a good idea to separate players by the team they are playing for
- This can be achieved by applying k-Nearest Neighbours, a lazy learning ML technique, to the bounding box detections of all players.
- To begin this process, we take all the bounding box detections for players from the output of the predict function using our finetuned YOLOv8 model
- Then for each frame, we apply our k-NN technique to each frame and assign team 1 and team 2 classes accordingly
-  

