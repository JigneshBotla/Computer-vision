import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r'C:\Users\botla\OneDrive\Desktop\Computer vision\PoseEstimationProjest\PoseVideos\dance.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Finished processing or could not read the frame.")
        break  

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c =img.shape
            print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv2.circle(img,(cx,cy),25,(255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = cv2.resize(img, (800, 600))

    cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
