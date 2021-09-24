import numpy as np
import cv2

class Mouse:
    def __init__(self, window_tytle):
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        self.neighbor = None
        cv2.setMouseCallback(window_tytle, self.callback, None)

    def callback(self, event, x, y, flags, param):
        self.mouseEvent["event"] = event
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["flags"] = flags

    def getEvent(self):
        return self.mouseEvent["event"]

    def getCoord(self):
        return [self.mouseEvent["x"], self.mouseEvent["y"]]

    def getNeighbor(self, corners):
        differ = np.abs(np.array(corners)-np.array(self.getCoord()))
        retsu_total = differ[:,:,0] + differ[:,:,1]
        min_index = retsu_total[:,0].argmin()
        self.neighbor = np.array(corners)[min_index]
        return self.neighbor