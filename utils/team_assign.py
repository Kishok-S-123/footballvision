import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster


player_img = cv2.imread("assets/sample_player_img.jpg")
player_img = cv2.cvtColor(player_img, cv2.COLOR_BGR2RGB)

class team_assign:

    def __init__(self, player_img: np.ndarray):
    
        self.player_img_height = np.shape(player_img)[0]
        self.player_img_width = np.shape(player_img)[1]

        shirt_img = player_img[:self.player_img_height//2,:,:]

        # plt.imshow(shirt_img)
        
        self.kmeans = sklearn.cluster.KMeans(n_clusters=2)

        self.flat_img = shirt_img.reshape((-1,3))

    def kmeans(self):
        self.kmeans.fit(self.flat_img)

        self.seg_img = self.kmeans.labels_.reshape((self.player_img_height//2, self.player_img_width))

        # plt.imshow(seg_img)

    def shirt_col(self):
        centroids = self.kmeans.cluster_centers_
        # based off assumption that the player will not take up the entire bounding box and thus the majority of the corners will be background
        corner_vals = [self.seg_img[0,0], self.seg_img[0,1], self.seg_img[1,0], self.seg_img[1,1]]
        common = max(set(corner_vals), key = corner_vals.count)
        self.background_col = centroids[common]
        self.player_col = centroids[common-1]

        # print(player_col)

