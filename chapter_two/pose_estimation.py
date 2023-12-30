import cv2
import mediapipe as mp
import time


"""
import the pose module from media pipe 
create a instance of the pose class 
Pose args: 
 - static_image_mode (boolean): If set to True, the pose detection is performed on each image, ideal for processing a batch of static, possibly unrelated, images
 - upper_body_only (boolean): When set to True, the model will detect and track only the upper body keypoints
 - smooth_landmarks (boolean): If True, it will smooth the landmarks across frames.
 - min_detection_confidence (float): This parameter controls the minimum confidence value ([0.0, 1.0]) for the person detection to be considered successful.
 - min_tracking_confidence (float): Similar to min_detection_confidence, but for the tracking process
 - model_complexity (int): This argument controls the complexity of the pose estimation model. ([0-2])

"""

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils





# get video from webcam 
cap = cv2.VideoCapture(0)

# set varaibles for fps 
pTime = 0
cTime = 0



while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        # draw landmarks 
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print (id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (0, 255, 0))


    



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