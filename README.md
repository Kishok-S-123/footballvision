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
- This can be achieved by applying k means, an unsupervised learning technique, to the bounding box detections of all players as we just need to split the players into their respective roles (team 1, team 2, referee).
- To begin this process, we take all the bounding box detections for players from the output of the predict function using our finetuned YOLOv8 model
- Then for each frame of the video, we can apply this technique to assign a team to each tracker id

- We begin by extracting a detection of a player from the results of the YOLO model
![alt text](assets/sample_player_img.jpg)

- We would like to use the player's shirt to assign teams, meaning that the bottom half of the image will not be required and may introduce unnecessary complexity
- As a result we shall crop the image to only include the top half of the player detection
![alt text](assets/player_top_half.png)

- With K-Means we are able to segment the image into 2 classes: player and background to a functional degree
![alt text](assets/player_seg_img.png)

- We can return our cluster centre values to see the average colour of the players kit, as well as the average colour values of the grass.
- This can then be placed into an RGB colour picker to see what our KMeans model has calculated:
![alt text](assets/playershirtcol.jpg)
![alt text](assets/grass_col.jpg)

- By doing this for every player detection in the video, we can achieve team identification, which can go a long way into gameplay analysis.


- Once we have the player shirt colour, we can compare 

## Video processing pipeline:
- Take in match footage
-  