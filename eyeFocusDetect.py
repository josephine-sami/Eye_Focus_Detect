import numpy as np  
import cv2  
import dlib
import queue
from scipy.spatial import distance as dist  

###dlib 68 feature
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
# FULL_POINTS = list(range(0, 68))  
# FACE_POINTS = list(range(17, 68))  
# JAWLINE_POINTS = list(range(0, 17))  
# RIGHT_EYEBROW_POINTS = list(range(17, 22))  
# LEFT_EYEBROW_POINTS = list(range(22, 27))  
# NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42)) 
LEFT_EYE_POINTS = list(range(42, 48))  
# MOUTH_OUTLINE_POINTS = list(range(48, 61))  
# MOUTH_INNER_POINTS = list(range(61, 68))  

#contanst
EYE_CLOSE_JUDGE = 0.2
EYE_CLOSE_CONTINUE = 600
COUNTER_LEFT = 0  
TOTAL_LEFT = 0  
COUNTER_RIGHT = 0  
TOTAL_RIGHT = 0
queue_1min=queue.Queue(maxsize=EYE_CLOSE_CONTINUE)
WINK_RATIO=0.66
DAZE_RATIO=0.96
"""
waitkey(1)==1ms
1min=60s=60000ms 
test==>waitkey(1)~~0.1s
1min wink 20 times 33%
""" 

## EAR function 
def eye_aspect_ratio(eye):  

    #vertical  ||
    A = dist.euclidean(eye[1], eye[5])  
    B = dist.euclidean(eye[2], eye[4])  
    #horizontal ---  
    C = dist.euclidean(eye[0], eye[3])  

    # compute the eye aspect ratio  
    ear = (A + B) / (2.0 * C)  
    return ear  


detector = dlib.get_frontal_face_detector()         #檢測器
predictor = dlib.shape_predictor(PREDICTOR_PATH)    #預測變量  

cap = cv2.VideoCapture(0)
#record#storePath='D:/Facial Expression/0605/'
#record#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#record#out = cv2.VideoWriter(storePath+"test.avi",fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()  
    if ret:  
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0) 
        #rext=[(251, 280) (509, 538)] rects=<dlib.dlib.rectangles object at 0x000001E5458E7090>
        for rect in rects: 
            x = rect.left()  
            y = rect.top()  
            #  x1 = rect.right()  
            #  y1 = rect.bottom()  
            landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])  

            left_eye = landmarks[LEFT_EYE_POINTS]  
            right_eye = landmarks[RIGHT_EYE_POINTS]  
            left_eye_hull = cv2.convexHull(left_eye)  
            right_eye_hull = cv2.convexHull(right_eye)  
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)  
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)  

            ear_left = eye_aspect_ratio(left_eye)  
            ear_right = eye_aspect_ratio(right_eye)  

            #print ear data
            cv2.putText(frame, "E.A.R. Left : {:.2f}".format(ear_left), (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  
            cv2.putText(frame, "E.A.R. Right: {:.2f}".format(ear_right), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            #print(COUNTER_LEFT,COUNTER_RIGHT)
            '''if ear_left < EYE_CLOSE_JUDGE:  
                COUNTER_LEFT += 1  
            else:  
                if COUNTER_LEFT >= EYE_CLOSE_CONTINUE:  
                    TOTAL_LEFT += 1  
                COUNTER_LEFT = 0  

            if ear_right < EYE_CLOSE_JUDGE:  
                 COUNTER_RIGHT += 1  
            else:  
                if COUNTER_RIGHT >= EYE_CLOSE_CONTINUE:  
                    TOTAL_RIGHT += 1  
                COUNTER_RIGHT = 0'''
            
            if queue_1min.full():
                queue_1min.get()
            if ear_left < EYE_CLOSE_JUDGE and ear_right < EYE_CLOSE_JUDGE:  
                COUNTER_RIGHT += 1  
                COUNTER_LEFT += 1 
                queue_1min.put(0)
            else:  
                if COUNTER_RIGHT >= EYE_CLOSE_CONTINUE and COUNTER_LEFT >= EYE_CLOSE_CONTINUE:  
                    TOTAL_RIGHT += 1 
                    TOTAL_LEFT += 1
                COUNTER_RIGHT = 0
                COUNTER_LEFT = 0
                queue_1min.put(1)
            
        #result   
        if COUNTER_RIGHT>=EYE_CLOSE_CONTINUE and COUNTER_LEFT >= EYE_CLOSE_CONTINUE:#close 60s or more
            cv2.putText(frame, "DON'T SLEEP !!!", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  
        if queue_1min.full() and sum(list(queue_1min.queue))<queue_1min.qsize()*WINK_RATIO:
            cv2.putText(frame, "Want To Sleep ???", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if queue_1min.full() and sum(list(queue_1min.queue))>queue_1min.qsize()*DAZE_RATIO:
            cv2.putText(frame, "IN A DAZE !", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        #print('open times',sum(list(queue_1min.queue)))
        #print(queue_1min.qsize,list(queue_1min.queue))
        
        #record#out.write(frame) 
        cv2.imshow("Are you focused?", frame)  
        
    per = 0xFF & cv2.waitKey(1)  
    if per == 27:  
        break 


#record#out.release()        
cap.release() 
cv2.destroyAllWindows()