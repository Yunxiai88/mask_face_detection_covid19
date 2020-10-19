import  os
import cv2
import numpy as np

from models.util import utils


class YoloDetector:

    def __init__(self):
        self.LABELS           = self.init_label()
        self.COLORS           = self.init_color()
        self.net, self.ln     = self.init_weight()

    # initial labels
    def init_label(self):
        labelsPath = utils.get_file_path("cfg", "classes.names")
        return open(labelsPath).read().strip().split("\n")

    # initial colors
    def init_color(self):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

    def init_weight(self):
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join(["data", "yolov4.weights"])
        configPath = os.path.sep.join(["cfg", "yolov4.cfg"])
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return (net, ln)

if __name__=='__main__':
    # load our YOLO object detector and determine only the *output* layer names
    print("[INFO] loading YOLO from disk begin...")
    yolo  = YoloDetector()
    print("[INFO] loading YOLO from disk end...")