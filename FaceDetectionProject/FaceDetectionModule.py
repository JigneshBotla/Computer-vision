import cv2
import mediapipe as mp
import time

class FaceDetector:
    def __init__(self, minDetectionConfidence=0.5):
        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        # print(self.results)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.fancyDraw(img,bbox)
                    self.mpDraw.draw_detection(img, detection)
                    cv2.putText(img, f'Confidence: {int(detection.score[0] * 100)}%', (bbox[0]-100, bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return img, bboxs
    



    def fancyDraw(self,img,bbox,l=30,t=10,rt=2):

        x,y,w,h=bbox
        x1,y1 = x+w,y+h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        #Top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)

        #Top right x1,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)

        #Bottom left x,y
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
        #Bottom right x1,y
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)


        return img


def main():
    video_path = r'C:\Users\botla\OneDrive\Desktop\Computer vision\FaceDetectionProject\videos\jog.mp4'

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    pTime = 0
    detector = FaceDetector(0.66)
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from video file")
            break

        img, bboxs = detector.findFaces(img)
        print(bboxs)

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

if __name__ == "__main__":
    main()
