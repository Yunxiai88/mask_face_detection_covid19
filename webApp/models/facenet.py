import os
import cv2
import csv
import time
import numpy as np

from PIL import Image
from mtcnn.mtcnn import MTCNN

from models.util import utils

from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

in_encoder = Normalizer('l2')
print("Using gpu: {0}".format(tf.test.is_gpu_available(
                            cuda_only=False,
                            min_cuda_compute_capability=None)))

class FaceNet:
    def __init__(self):
        self.CONFIDENCE       = 0.5
        self.THRESHOLD        = 0.3
        self.LABELS           = self.init_label()
        self.COLORS           = self.init_color()
        self.net              = self.init_weight()
        self.model_Face       = self.init_model()
        self.database         = self.init_database()

    # function to read database
    def init_database(self):
        print("read data from database...")
        database = {}

        dbPath = utils.get_file_path("data", "dict.csv")
        reader = csv.reader(open(dbPath), delimiter='\n')

        for row in reader:
            data = row[0].split(",", 1)
            encode = data[1].replace('"','')
            encode = encode.replace('[','')
            encode = encode.replace(']','')
            encode = np.fromstring(encode, dtype=float, sep=',')
            database[data[0]] = encode
        return database

    def init_model(self):
        print("initial facenet model...")
        facePath = utils.get_file_path("data", "facenet_keras.h5")
       # cvPath = os.path.sep.join(["data", "haarcascade_frontalface_alt2.xml"])
        #faceCascade = cv2.CascadeClassifier(cvPath)
        return load_model(facePath, custom_objects={ 'loss': self.triplet_loss }, compile=False)

    # initial labels
    def init_label(self):
        print("initial yolo label...")
        labelsPath = utils.get_file_path("cfg", "classes.names")
        return open(labelsPath).read().strip().split("\n")

    # initial colors
    def init_color(self):
        print("initial yolo colors...")
        np.random.seed(42)
        return np.random.randint(0, 255, size=(len(self.LABELS), 3), dtype="uint8")

    def init_weight(self):
        print("[INFO] loading YOLO from disk...")
        # derive the paths to the YOLO weights and model configuration
        weightsPath = utils.get_file_path("data", "yolov4.weights")
        configPath = utils.get_file_path("cfg", "yolov4.cfg")
        return cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def triplet_loss(y_true, y_pred, alpha = 0.2):
        print("using triplet_loss function...")
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

        # Step 1: Compute the (encoding) distance between the anchor and the positive
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
        # Step 2: Compute the (encoding) distance between the anchor and the negative
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
        # Step 3: subtract the two previous distances and add alpha.
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
        loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

        return loss

    # get face embedding and perform face recognition
    def get_embedding(self, image):
        print("get enbedding code function begin...")
        # scale pixel values
        face = image.astype('float32')
        # standardization
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        face = cv2.resize(face,(160,160))
        face = np.expand_dims(face, axis=0)
        encode = self.model_Face.predict(face)[0]
        return encode

    def find_person(self, encoding, min_dist=1):
        print("find person function begin...")
        min_dist = float("inf")
        encoding = in_encoder.transform(np.expand_dims(encoding, axis=0))[0]
        for (name, db_enc) in self.database.items():
            dist = cosine(db_enc, encoding)
            if dist < 0.5 and dist < min_dist:
                min_dist = dist
                identity = name

        if min_dist > 0.5:
            return "None"
        else:
            return identity
        return "None"

    def detect(self, frame, W, H):
        # construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        layer = self.net.getLayerNames()
        layer = [layer[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # calculate execute time
        start = time.time()
        layerOutputs = self.net.forward(layer)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        #faces_list = []
        #encodes = []
        names = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.CONFIDENCE:
                    # scale the bounding box coordinates back relative to
                    # the size of the image
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    if classID == 0 :
                        classIDs.append(classID)
                        names.append('')
                    elif classID == 1:
                        #openCV
                        #convert to greyscale
                        #faces_list=[]
                        #encodes=[]
                        crop = frame[y:y+int(height), x:x+int(width)]

                        #gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        #detect face
                        #faces = faceCascade.detectMultiScale(crop,
                                                            #scaleFactor=1.1,
                                                            #minNeighbors=5,
                                                            ##minSize=(60, 60),
                                                            #flags=cv2.CASCADE_SCALE_IMAGE)
                        #to draw rectangle
                        label = ""
                        #for (x, y, w, h) in faces:
                        #face_frame = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        face_frame = img_to_array(crop)
                        name = "None"
                        if face_frame.size != 0 :
                            face_frame = cv2.resize(face_frame, (160, 160))
                            encode = self.get_embedding(face_frame)
                            name = self.find_person(encode)
                        if name == "None":
                            label = "Not found"
                        else :
                            label = name

                        classIDs.append(classID)
                        names.append(label)

        # apply non-maximal suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE, self.THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]]+":"+names[i], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 10)

    # use MTCNN to detect faces and return face array
    def extract_mtcnn_face(self, filename, required_size=(160, 160)):
        print("extracting face from image")
        detector = MTCNN()

        image = Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        return face_array

    # encode person and save into db
    def save_encode_db(self, label, filename):
        print("encoding was begining for: " + label)

        # extract face
        imagePath = os.path.sep.join(["uploads", filename])
        face_frame = self.extract_mtcnn_face(imagePath)

        # get enbedding code
        self.database[label] = self.get_embedding(face_frame)

        # write into db
        dbPath = os.path.sep.join(["data", "dict.csv"])
        with open(dbPath, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.database.items():
               value = list(value)
               writer.writerow([key, value])


