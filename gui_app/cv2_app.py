import os
import sys
import time

import cv2
import numpy as np
from numpy.core.numeric import Inf
from natsort import natsorted

from cv2_gadget import Mouse

SIGNAL_ENTER = 13
SIGNAL_ESC   = 27
SIGNAL_CLICK_ON = cv2.EVENT_LBUTTONDOWN
fINF = float('inf')

def draw_contour(img, cnt, epsilon_rate=0.0001, color=(0,0,255), fill_option=1):
    approx = cv2.approxPolyDP(cnt, epsilon_rate*cv2.arcLength(cnt, True), True)
    img = cv2.drawContours(img, [approx], 0, color, fill_option)
    return img

def contour_ditector(gray_img, epsilon_rate=0.0001, err_permission=15):
    selected_contours = []
    contours, _ = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < err_permission:
            pass
        else:
            rgb_img = draw_contour(rgb_img, cnt)
            selected_contours.append(cnt)
    return rgb_img, selected_contours

def show_info(img, msg, org, 
              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1.0,
              color=(255, 255, 255),
              thickness=1):
    cv2.putText(img, msg, org, fontFace, fontScale, color, thickness)

if __name__=="__main__":
    #初期設定
    win_name = "indicated"
    img_path = "C:/Users/ryota/Desktop/MRI_prj/subject01/pose01"
    #img_path = "/Users/ryotaro/Desktop/MRI_prj/subject01/pose01"

    #ディレクトリ内の画像ファイルを取得
    png_files = natsorted([i for i in os.listdir(img_path) if ".png" in i])

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    mouse = Mouse(win_name) 
    for png in png_files:
        #輪郭抽出
        bgr_img = cv2.imread(img_path+"/"+png, 0)
        _, th1 = cv2.threshold(bgr_img, 15, 255,cv2.THRESH_BINARY)
        indicated_img, contours = contour_ditector(th1)
        if len(contours) == 0:
            continue
        while True:
            show_info(indicated_img, str(png), (3,10), fontScale=0.4)
            show_info(indicated_img, "cnt : {0}".format(len(contours)), (3,20), fontScale=0.4)
            cv2.imshow(win_name, indicated_img)
            if mouse.getEvent() == SIGNAL_CLICK_ON:
                #クリックした場所から最も近い輪郭配列のインデックスを取得
                clicked_point = mouse.getCoord()
                min_cluster, min_clusterAddr = None, fINF
                for i, cnt in enumerate(contours):
                    try:
                        row, _, column = cnt.shape
                    except:
                        print("cnt shape is abnormality")
                        cv2.destroyWindow(win_name)
                        sys.exit()
                    cnt = cnt.reshape((row, column))
                    diff = cnt - clicked_point
                    distance = np.sqrt(diff[:,0]**2 + diff[:,1]**2)
                    cnt_ShortestDistance = np.argmin(distance)
                    if min_clusterAddr > cnt_ShortestDistance:
                        min_cluster, min_clusterAddr = i, cnt_ShortestDistance
                
                #クリック箇所の色付け
                rep_cnt = contours[min_cluster]
                indicated_img = draw_contour(indicated_img, rep_cnt, color=(255,0,0), fill_option=-1)
            
            key = cv2.waitKey(1) & 0xFF
            if key == SIGNAL_ESC:
                cv2.destroyWindow(win_name)
                sys.exit()
            elif key == SIGNAL_ENTER:
                break
