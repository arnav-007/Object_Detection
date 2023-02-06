import cv2
net =cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416,416), scale=1/255)
#load class lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name =class_name.strip()
        classes.append(class_name)

        print("object list")
        print(classes)
#initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1288)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 728)

# create  window
# cv2.namedWindow("frame")
# cv2.setMouseCallback("frame",click_button)
while True:
    #get frames
    ret, frame = cap.read()
   #object detection
    (class_ids,scores,bboxes)=model.detect(frame)
    for class_id, score, bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bbox
       # print(x,y,w,h)
        class_name =classes[class_id]
        cv2.putText(frame, class_name, (x,y-10), cv2.FONT_HERSHEY_PLAIN,2,(200, 0, 50), 2)
        cv2.rectangle(frame,(x,y),(x+w, y+h),(200, 0, 50), 3)

        #create button
        cv2.rectangle(frame, (28, 28), (228,78), (0, 0, 288), -1)

        cv2.putText(frame, "person", (38,68), cv2.FONT_HERSHEY_PLAIN,3,(255, 255, 255), 2)

    #print("class ids",class_ids)
    #print("scores",scores)
    #print("bboxes",bboxes)



    cv2.imshow("Frame",frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break



