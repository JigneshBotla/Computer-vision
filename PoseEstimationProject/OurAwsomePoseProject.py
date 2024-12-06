import cv2
import time
import PoseModule as pm


# cap = cv2.VideoCapture(r'C:\Users\botla\OneDrive\Desktop\Computer vision\PoseEstimationProjest\PoseVideos\jog.mp4')
cap = cv2.VideoCapture(0)
pTime=0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[0])  
    
    cv2.circle(img,(lmList[0][1],lmList[0][2]),15,(0,0,255),cv2.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
