#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import pyttsx3
friend=pyttsx3.init()

yolo = cv2.dnn.readNet("yolov4.cfg", "yolov4.weights")

classes = []
with open("coco.names",'r') as f:
    classes = f.read().splitlines()


#get layers of the network
layer_names = yolo.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]
print("YOLO LOADED")


def upload_file():
    global img
    global filename
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img = ImageTk.PhotoImage(file=filename)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=4,column=1)

def classify_object():
    img = cv2.imread(filename)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
     swapRB=True, crop=False)
    #Detecting objects
    yolo.setInput(blob)
    outs = yolo.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
			1/2, color, 2)

    cv2.imshow("Image",img)
    friend.say(label)
    friend.runAndWait()
    cv2.waitKey(0)


def classify_object2():
    video_capture = cv2.VideoCapture(0)
    while True:
        
# Capture frame-by-frame
        re,img = video_capture.read()
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    
        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
         swapRB=True, crop=False)
        #Detecting objects
        yolo.setInput(blob)
        outs = yolo.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        #We use NMS function in opencv to perform Non-maximum Suppression
        #we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
                1/2, color, 2)

        cv2.imshow("Image",img)
        #friend.say(label)
        #friend.runAndWait()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

'''def classify_object1():
    video_capture = cv2.VideoCapture(0)
    while True: 
# Capture frame-by-frame
        r,img = video_capture.read()
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # USing blob function of opencv to preprocess image
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
         swapRB=True, crop=False)
        #Detecting objects
        yolo.setInput(blob)
        outs = yolo.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        #We use NMS function in opencv to perform Non-maximum Suppression
        #we give it score threshold and nms threshold as arguments.
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
                            1/2, color, 2)

        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()'''

my_w = tk.Tk()
my_w.geometry("400x500")  # Size of the window
my_w['bg']='black'
my_w.title('Object Identifier')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Upload Picture and Detect Object',width=30,font=my_font1, fg='black', bg='orange')  
l1.grid(row=1,column=1)
b1 = tk.Button(my_w, text='Upload File', 
   width=20,command = lambda:upload_file(),fg='black', bg='orange')
b1.grid(row=2,column=1)
b2 = tk.Button(my_w, text='Submit', width=20,command = lambda:classify_object(), fg='black', bg='orange')
b2.grid(row=3,column=1) 
#b3 = tk.Button(my_w, text='Live', width=20,command = lambda:classify_object1(), fg='black', bg='orange')
#b3.grid(row=4,column=1) 




my_w.mainloop()  # Keep the window open


    



# In[ ]:




