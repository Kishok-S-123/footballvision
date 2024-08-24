# footballvision
yolov8 vision based football analysis bot using Bundesliga Videos dataset from Kaggle competition: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout

## Project Goals:
- YOLO object detection- detection of ball and people only involved in the game i.e. players, referee, goalie, etc
- Player tracking- keeping track of each player (assign an id for each player and keeping track of location)
- Player colour assignment- detection of team based on shirt colour
- ball interpolation- interpolating position of ball in frames where it is not picked up by yolo
- camera movement estimator (optical flow)- keeping track of the movement of the camera in order to calculate distances travelled etc
- speed and distance estimator- estimating speed of each player as well as total distance travelled