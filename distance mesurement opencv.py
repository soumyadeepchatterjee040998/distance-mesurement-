import cv2
import numpy as np
import time
from scipy.spatial import distance as dist

Tsize = 320
prob_tresh = 0.5
nms_threshold = 0.2

ModelConfigFile = "yolov3.cfg"
ModelWeightFile = "yolov3.weights"
model = cv2.dnn.readNetFromDarknet(ModelConfigFile,ModelWeightFile)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
classfile= "coco.names"
with open(classfile,'r') as f:
    classnames = f.read().rsplit('\n')
color = (0,255,0)
MIN_DISTANCE = 500
def findbox(outputs,img,cent):
    global color,MIN_DISTANCE
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    prob = []
    for output in outputs:
        for det in output:
            rest = det[5:]
            classId = np.argmax(rest)
            conf = rest[classId]
            if conf>= prob_tresh:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int(det[0]*wT-w/2),int(det[1]*hT-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                prob.append(float(conf))
    indices = cv2.dnn.NMSBoxes(bbox,prob,prob_tresh,nms_threshold=nms_threshold)
    person_indices = []
    person_prob = []
    for index in indices:
        classId = classIds[index[0]]
        if classnames[classId] == 'person':
            x,y,w,h = bbox[index[0]][0],bbox[index[0]][1],bbox[index[0]][2],bbox[index[0]][3]
            person_indices.append(index)
            person_prob.append(prob[index[0]])
            cent.add((int(x+(w/2)),int(y+(h/2))))           
    if len(person_indices)>=2:
            D = dist.cdist(np.array(list(cent)),np.array(list(cent)),metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < MIN_DISTANCE:
                        violate.add(i)
                        violate.add(j)              
    try:
        for i,index in enumerate(person_indices):
            index = index[0]
            x,y,w,h = bbox[index][0],bbox[index][1],bbox[index][2],bbox[index][3]
            conf = int(person_prob[index]*100)
            p = (int(x+(w/2)),int(y+(h/2)))
            if i in violate:
                color = (0,0,255)
            else:
                color = (0,255,0)                
            cv2.circle(img,p,5,color,-1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(img,f'PERSON : {int(person_prob[index]*100)}%',(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255, 0, 255),2)
    except:
        return   
cap = cv2.VideoCapture(0)
cv2.namedWindow("output")
time.sleep(3)
while(cap.isOpened()):
    flag,img = cap.read()
    if flag==False:
        break
    else:
        cent = set()
        violate = set()
        blob = cv2.dnn.blobFromImage(img,1/255,(Tsize,Tsize),[0,0,0],1,crop=False)
        outputlayers = model.getUnconnectedOutLayersNames()
        model.setInput(blob)
        outputs = model.forward(outputlayers)
        findbox(outputs,img,cent)
        cv2.imshow("output",img)
        k = cv2.waitKey(1)
        if k==27:
            break
cap.release()
cv2.destroyAllWindows()