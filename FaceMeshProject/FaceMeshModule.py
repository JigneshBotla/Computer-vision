import cv2
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon,
            min_tracking_confidence=self.minTrackCon
        )
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        faceLms,
                        self.mpFaceMesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.drawSpec,
                        connection_drawing_spec=self.drawSpec
                    )
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append((id, x, y))
                faces.append(face)
        return img, faces
                




def main():
    video_path = r'C:\Users\botla\OneDrive\Desktop\Computer vision\FaceMeshProject\videos\jog.mp4'

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    pTime = 0

    detector = FaceMeshDetector()


    while True:

        success, img = cap.read()
        img,faces= detector.findFaceMesh(img)
        if not success:
            print("Error: Could not read frame from video file")
            break
        if len(faces)!=0:
            print(len(faces))
        cv2.putText(img, f'Faces : {str(len(faces))}', (20,20 ), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

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


if __name__=="__main__":
    main()