import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    # save the frame of the video cap to img 
    success, img = cap.read()

    # hand tracking needs image to be in RGB convert it from BGR 
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get the landmarks on to the hand using the mpHands.Hands() function
    results = hands.process(imageRGB)

    if results.multi_hand_landmarks:
        # loops through all the hand marks as handLms
        for handLms in results.multi_hand_landmarks:
            # get the id of the landmark, and land mark x and y 
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, c)

                
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


            # draws the handmarks and connect the line using the position of the hand coming results.multi_hand_landmarks 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    # get the fps 
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    # present the fps 
    # param: img, fps, size, font, size, color, weight 
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()