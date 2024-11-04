import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


player_img = cv2.imread("assets/sample_player_img.jpg")
player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)

class team_assign:

    def __init__(self, frame: np.ndarray, player_boxes):

    
        pass

    
    def get_sub_image(frame, box):
        return frame[int(box[0][1]):int(box[0][3]) , int(box[0][0]):int(box[0][2]) ,:]

    def get_player_col(self, player_img):

        player_img_height = np.shape(player_img)[0]
        player_img_width = np.shape(player_img)[1]

        shirt_img = player_img[:player_img_height//2,:,:]
        
        # plt.imshow(shirt_img)
        
        kmeans = sklearn.cluster.KMeans(n_clusters=2)

        flat_img = shirt_img.reshape((-1,3))
        
        kmeans.fit(flat_img)

        seg_img = self.kmeans.labels_.reshape((player_img_height//2, player_img_width))
        
        # plt.imshow(seg_img)

        centroids = kmeans.cluster_centers_
        # based off assumption that the player will not take up the entire bounding box and thus the majority of the corners will be background
        corner_vals = [seg_img[0,0], seg_img[0,1], seg_img[1,0], seg_img[1,1]]
        common = max(set(corner_vals), key = corner_vals.count)
        background_col = centroids[common]
        player_col = centroids[common-1]

        return player_col 

        # print(player_col)


    def team_cols(self, player_boxes, frame):
        # taking shirt colour from each of the players in the frame and adding to list
        shirt_cols = []
        for player in player_boxes:
            player_img = self.get_sub_image(frame, player)
            shirt_col = self.get_player_col(player_img)
            shirt_col.append(shirt_col)


        # perform kmeans on shirt colours 

        colour_kmeans = sklearn.cluster.KMeans(n_clusters = 2)
        colour_kmeans.fit(shirt_cols)

        self.team_colours[]

        