import cv2
import mediapipe as mp
import time

video_path = r'C:\Users\botla\OneDrive\Desktop\Computer vision\FaceDetectionProject\videos\jog.mp4'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

pTime = 0


mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    
    if not success:
        print("Error: Could not read frame from video file")
        break

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results= faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih),int(bboxC.width * iw),int(bboxC.height * ih)
            cv2.rectangle(img,bbox,(255,0,255),3)
            cv2.putText(img, f'Confidence: {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = cv2.resize(img, (800, 600))

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
