import cv2
import mediapipe as mp
import time






class poseDetector():
    def __init__(self, mode=False, modelComp=1, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.modelComp = modelComp
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.modelComp, self.upBody, self.smooth,
                                     self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils






    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks and draw :
            # draw landmarks 
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
    
    
        return img
    

    def getPosition(self, img, draw=True):
        lmList = [ ]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0))
            
        return lmList


def main():
    cap = cv2.VideoCapture(0)

    # set varaibles for fps 
    pTime = 0
    cTime = 0
    
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img, False)

        print(lmList)
    


        # get the fps 
        cTime = time.time()
        fps = 1/(cTime - pTime)

        # present the fps 
        # param: img, fps, location, font, size, color, weight 
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)


        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()





if __name__ == '__main__':
    main()