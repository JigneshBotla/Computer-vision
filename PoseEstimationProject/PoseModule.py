import cv2
import mediapipe as mp
import time

class poseDetector:
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=1,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        img = cv2.resize(img, (800, 600))
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

# Example usage
if __name__ == "__main__":
    # cap = cv2.VideoCapture(r'C:\Users\botla\OneDrive\Desktop\Computer vision\PoseEstimationProjest\PoseVideos\jog.mp4')
    cap = cv2.VideoCapture(0)
    detector = poseDetector()
    pTime=0

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

    cap.release()
    cv2.destroyAllWindows()
