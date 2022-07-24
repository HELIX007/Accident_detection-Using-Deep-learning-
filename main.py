from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import time
import cv2
import tensorflow.compat.v1 as tf
from collections import namedtuple
from collections import defaultdict
from io import StringIO
from PIL import Image
import numpy as np
import winsound
import imutils

main = tkinter.Tk()
main.title("Accident Detection")
main.geometry("1300x1200")

net = cv2.dnn.readNetFromCaffe("model/MobileNetSSD_deploy.prototxt.txt", "model/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

global filename
global detectionGraph
global msg


def loadModel():
    global detectionGraph
    detectionGraph = tf.Graph()
    with detectionGraph.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile('model/frozen_inference_graph.pb', 'rb') as file:
            serializedGraph = file.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')

    messagebox.showinfo("Training model loaded", "Training model loaded")


def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


def uploadVideo():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n");


def calculateCollision(boxes, classes, scores, image_np):
    global msg
    # cv2.putText(image_np, "NORMAL!", (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    for i, b in enumerate(boxes[0]):
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
            if scores[0][i] > 0.5:
                for j, c in enumerate(boxes[0]):
                    if (i != j) and (classes[0][j] == 3 or classes[0][j] == 6 or classes[0][j] == 8) and scores[0][
                        j] > 0.5:
                        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
                        ra = Rectangle(boxes[0][i][3], boxes[0][i][2], boxes[0][i][1], boxes[0][i][3])
                        rb = Rectangle(boxes[0][j][3], boxes[0][j][2], boxes[0][j][1], boxes[0][j][3])
                        ar = rectArea(boxes[0][i][3], boxes[0][i][1], boxes[0][i][2], boxes[0][i][3])
                        col_threshold = 0.6 * np.sqrt(ar)
                        area(ra, rb)
                        if (area(ra, rb) < col_threshold):
                            print('accident')
                            msg = 'ACCIDENT!'
                            beep()
                            return True
                        else:
                            return False


def rectArea(xmax, ymax, xmin, ymin):
    x = np.abs(xmax - xmin)
    y = np.abs(ymax - ymin)
    return x * y


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    return dx * dy


def detector():
    global msg
    msg = ''
    cap = cv2.VideoCapture(filename)
    with detectionGraph.as_default():
        with tf.Session(graph=detectionGraph) as sess:
            while True:
                ret, image_np = cap.read()
                (h, w) = image_np.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
                net.setInput(blob)
                detections = net.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.2:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        if (confidence * 100) > 50:
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            cv2.rectangle(image_np, (startX, startY), (endX, endY), COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detectionGraph.get_tensor_by_name('image_tensor:0')
                boxes = detectionGraph.get_tensor_by_name('detection_boxes:0')
                scores = detectionGraph.get_tensor_by_name('detection_scores:0')
                classes = detectionGraph.get_tensor_by_name('detection_classes:0')
                num_detections = detectionGraph.get_tensor_by_name('num_detections:0')
                if image_np_expanded[0] is not None:
                    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
                                                                        feed_dict={image_tensor: image_np_expanded})
                    calculateCollision(boxes, classes, scores, image_np)
                    cv2.putText(image_np, msg, (230, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.imshow('Accident Detection', image_np)
                    if cv2.waitKey(5) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break


def exit():
    main.destroy()


font = ('times', 16, 'bold')
title = Label(main, text='Accident Detection')
title.config(bg='light cyan', fg='pale violet red')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Load & Generate CNN Model", command=loadModel)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='light cyan', fg='pale violet red')
pathlabel.config(font=font1)
pathlabel.place(x=460, y=100)

webcamButton = Button(main, text="Browse System Videos", command=uploadVideo)
webcamButton.place(x=50, y=150)
webcamButton.config(font=font1)

webcamButton = Button(main, text="Start Accident Detector", command=detector)
webcamButton.place(x=50, y=200)
webcamButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=330, y=250)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
text.config(font=font1)

main.config(bg='snow3')
main.mainloop()