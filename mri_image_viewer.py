import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def get_contours_xyz(gray_img, z_pos, epsilon_rate=0.0001, err_permission=15):
    contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordination = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < err_permission:
            pass
        else:
            x = cnt[:,:,0][:,0]
            y = cnt[:,:,1][:,0]
            z = np.full_like(x, z_pos)
            coordination.append([x, y, z])
    return coordination

if __name__=="__main__":
    threshold = 15
    data_path = "C:/Users/ryota/Desktop/MRI_dataset/subject01"
    pose      = "/pose01"
    contents  = os.listdir(data_path + pose)
    images_list = [f for f in contents if (os.path.isfile(os.path.join(data_path + pose, f)) and f.endswith('.png'))]
    images_list = images_list[50:100]

    layers_list = []
    for i, img in enumerate(images_list):
        to_img = data_path + pose + "/" + img
        bgr_img = cv2.imread(to_img, 0)
        _, gray_img = cv2.threshold(bgr_img, threshold, 255,cv2.THRESH_BINARY)
        contours_coordination = get_contours_xyz(gray_img, i)
        layers_list.append(contours_coordination)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("x", size = 14)
    ax.set_ylabel("y", size = 14)
    ax.set_zlabel("z", size = 14)

    ax.set_xlim(100, 500)
    ax.set_ylim(0,   500)
    ax.set_zlim(-10, len(images_list))

    for layer in layers_list:
        for cnt in layer:
            x = cnt[0].tolist()
            y = cnt[1].tolist()
            z = cnt[2].tolist()
            ax.add_collection3d(Poly3DCollection([list(zip(x,y,z))], color='r',alpha=0.))
    plt.show()