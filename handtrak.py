import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
hhands=mp.solutions.hands
hands = hhands.Hands()
mpDrwa = mp.solutions.drawing_utils

#frame per second
prvioustime = 0
currenttime = 0

while True:
    ret,img= cap.read()
    imgBGR=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgBGR)
    print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in  enumerate(handlms.landmark): 
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int (lm.x*w),int(lm.y*h)
                print(id,cx,cy)  
                cv2.circle(img,(cx,cy),8,(255,0,255),cv2.FILLED)
            mpDrwa.draw_landmarks(img,handlms,hhands.HAND_CONNECTIONS)
    currenttime=time.time()
    fps = 1/(currenttime-prvioustime)
    prvioustime = currenttime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==13:
        break
cv2.destroyAllWindows()