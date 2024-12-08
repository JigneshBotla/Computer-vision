import cv2
import mediapipe as mp
import time

video_path = r'C:\Users\botla\OneDrive\Desktop\Computer vision\FaceMeshProject\videos\jog.mp4'

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2,min_detection_confidence=0.75)
drawSpec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from video file")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                img,
                faceLms,
                mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawSpec,
                connection_drawing_spec=drawSpec
            )
            for id,lm in enumerate(faceLms.landmark):
                # print(lm)
                ih,iw,ic=img.shape
                x,y = int(lm.x*iw),int(lm.y*ih)
                print(id,x,y)

                

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
