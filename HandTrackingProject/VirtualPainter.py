import cv2
import numpy as np
import os
import HandTrackingModule as htm

# Constants for brush and eraser thickness
brushThickness = 15
eraserThickness = 50

# Load header images
folderPath = r'C:\Users\botla\OneDrive\Desktop\Computer vision\HandTrackingProject\Header'
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]  # Default header image
drawColor = (0, 255, 255)  # Default draw color

header_height, header_width, _ = header.shape
print(f"Header dimensions: {header.shape}")

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  # Set width
cap.set(4, 850)   # Set height

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0  # Previous points for drawing

# Initialize a blank canvas
imgCanvas = None

while True:
    # 1. Capture the frame
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # Flip the frame horizontally

    # Initialize the canvas dynamically based on frame size
    height, width, _ = img.shape
    if imgCanvas is None or imgCanvas.shape[:2] != (height, width):
        imgCanvas = np.zeros((height, width, 3), np.uint8)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. Selection mode (two fingers up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('Selection mode')

            # Check for clicks in the header area
            if y1 < header_height:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (0, 255, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (128, 128, 128)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 255)
                elif 1200 < x1 < 1400:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)
            
            # Draw a rectangle for selection feedback
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. Drawing mode (index finger up)
        elif fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('Drawing mode')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            # Draw on canvas
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    # Merge canvas with the frame
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Resize and set header
    resized_header = cv2.resize(header, (width, header_height))
    img[0:header_height, 0:width] = resized_header

    # Display the frames
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
