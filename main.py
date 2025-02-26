from pickle import TRUE
from urllib import response
import cv2 as cv
from matplotlib.pyplot import pie
import mediapipe as mp
import time
import utils, math
import numpy as np
import serial

try:
    py_serial = serial.Serial('COM5',9600)
    if py_serial.readable():
        print("I can read serial")
except:
    pass


blinkTimeArray = []
blinkTimeArray.append(0)
preBlinkIndex = 0
blinkIndex = 0
startBool = False

# variables 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
n1 = 0
# constants
#CLOSED_EYES_FRAME =3
CLOSED_EYES_FRAME = 1
FONTS =cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 
camera = cv.VideoCapture(0)
# landmark detection function 
def landmarksDetection(img, results, draw=True):  ## draw = True : draw each point in the face
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....] : len(mesh_coord) = 468
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]  # multi_face_landmarks : {x y z}    ##.landmark => landmark 부분
   
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]    ## circle the mesh_point

    
    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #print("rh_right : ", rh_right)   ## (x, y) 좌표로 나옴
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)


    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    
    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    #cv.imshow('mask', mask)    ## 눈만 흰색으로 표시됨.
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images    ## 저주파 필터래.  ## 눈으로 볼 때 두 값의 차이는 안 느껴져.
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)   ## kernel size
    #cv.imshow('gaussian_blur',gaussain_blur)
    #cv.imshow('median_blur', median_blur)

    # applying thrsholding to convert binary_image
    #ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)   ## 눈 pixel 밝기 값이 130보다 크면 255로
    ret, threshed_eye = cv.threshold(median_blur, 80, 255, cv.THRESH_BINARY)    ## 같은 눈동자에서도 pixel 밝기 값의 차이가 있어.
    ret, threshed_eye = cv.threshold(median_blur, 90, 255, cv.THRESH_BINARY)    ## 눈동자 pixel 밝기가 0이 아니네 (완전 검정이 아닌가봐)
    ret, threshed_eye = cv.threshold(median_blur, 50, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)  ## 어두운 곳일 수록 기준 값 낮춰야 해. (다른 어두운 부분이 검정 됨)
                                                                               ## 고개 각도, 얼굴을 정면에서 보냐 측면에서 보냐 따라 검정 maximum 기준 달라져
                                                                                ## 정면 응시가 눈의 정확히 중앙에 오지 않네.
                                                                                ## => 오, 왼 기준 값 적게 잡기.
    
    #threshed_eye = cv.adaptiveThreshold(median_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15, -2) 
    
    
    # create fixd part for eye with 
    piece = int(w/3)     
    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]    # [[255 255 255 ...]
                                                #  [255 255 255 ...]
                                                #  [255 255 0   0  ]...]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece+piece :w]
    
    
    piece1 = int(h/3)

    up_piece = threshed_eye[0:piece1, 0:w]
    down_piece = threshed_eye[piece1+piece1:h,0:w]
    
    # 눈 중앙 흰 부분 되면 up 되도록 눈 중앙 piexel 값들 받아옴.
    #white_center_piece =threshed_eye[piece1:piece1*2,piece:piece*2]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece, up_piece)


    return eye_position, color 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece, fourth_piece):  # type(first_piece) : class 'numpy.ndarray'   
    # counting black pixel in each part                   
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)

    
    up_part = np.sum(fourth_piece==0)
    
    # //new trial part    눈 위치 위 아래 먼저 판단 후, 좌 우 판단 (위로 해도 오,왼으로 판단되는 경우 있어서.)
    eye_parts = [up_part, center_part]
    max_index = eye_parts.index(max(eye_parts))
    if max_index == 0:
        pos_eye = "UP"
        color = [utils.GRAY, utils.PURPLE]
        sending_char = 'u'
    else :
        
    
        # creating list of these values
        eye_parts = [right_part, center_part, left_part]

        # getting the index of max values in the list 
        max_index = eye_parts.index(max(eye_parts))
        pos_eye ='' 
        if max_index==0:
            pos_eye="RIGHT"
            sending_char = 'r'
            color=[utils.BLACK, utils.GREEN]
            if startBool == True:
                try:
                    py_serial.write(sending_char.encode())
                except:
                    pass 
        elif max_index==1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]
            sending_char = 'c'
            if startBool == True:
                try:
                    py_serial.write(sending_char.encode())
                except:
                    pass 
        elif max_index ==2:
            pos_eye = 'LEFT'
            sending_char = 'l'
            color = [utils.GRAY, utils.YELLOW]
            if startBool == True:
                try:
                    py_serial.write(sending_char.encode())
                except:
                    pass
        
    return pos_eye, color

# Eyes Postion Estimator 
def positionEstimator_left(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images    ## 저주파 필터래.  ## 눈으로 볼 때 두 값의 차이는 안 느껴져.
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)   ## kernel size
    #cv.imshow('gaussian_blur',gaussain_blur)
    #cv.imshow('median_blur', median_blur)

    # applying thrsholding to convert binary_image
    #ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)   ## 눈 pixel 밝기 값이 130보다 크면 255로
    ret, threshed_eye = cv.threshold(median_blur, 80, 255, cv.THRESH_BINARY)    ## 같은 눈동자에서도 pixel 밝기 값의 차이가 있어.
    ret, threshed_eye = cv.threshold(median_blur, 90, 255, cv.THRESH_BINARY)    ## 눈동자 pixel 밝기가 0이 아니네 (완전 검정이 아닌가봐)
    ret, threshed_eye = cv.threshold(median_blur, 50, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)  ## 어두운 곳일 수록 기준 값 낮춰야 해. (다른 어두운 부분이 검정 됨)
    threshed_eye = cv.adaptiveThreshold(median_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15, -2)                                                                            ## 고개 각도, 얼굴을 정면에서 보냐 측면에서 보냐 따라 검정 maximum 기준 달라져
                                                                                ## 정면 응시가 눈의 정확히 중앙에 오지 않네.
                                                                                ## => 오, 왼 기준 값 적게 잡기.
    cv.imshow('threshed_eye',threshed_eye)    
    # create fixd part for eye with 
    piece = int(w/3)     
    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]    # [[255 255 255 ...]
                                                #  [255 255 255 ...]
                                                #  [255 255 0   0  ]...]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece+piece :w]
    piece1 = int(h/3)

    up_piece = threshed_eye[0:piece1, 0:w]
    down_piece = threshed_eye[piece1+piece1:h,0:w]
    
    # 눈 중앙 흰 부분 되면 up 되도록 눈 중앙 piexel 값들 받아옴.
    #white_center_piece =threshed_eye[piece1:piece1*2,piece:piece*2]
    
    # calling pixel counter function
    eye_position, color = pixelCounter_left(right_piece, center_piece, left_piece, up_piece)

    return eye_position, color 

def pixelCounter_left(first_piece, second_piece, third_piece, fourth_piece):  
    # counting black pixel in each part                     각각의 눈에 적용
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)

    
    up_part = np.sum(fourth_piece==0)
    
    # //new trial part    눈 위치 위 아래 먼저 판단 후, 좌 우 판단 (위로 해도 오,왼으로 판단되는 경우 있어서.)
    eye_parts = [up_part, center_part]
    max_index = eye_parts.index(max(eye_parts))
    if max_index == 0:
        pos_eye = "UP"
        color = [utils.GRAY, utils.PURPLE]
        sending_char = 'u'
    else :
        # creating list of these values
        eye_parts = [right_part, center_part, left_part, up_part]

        # getting the index of max values in the list 
        max_index = eye_parts.index(max(eye_parts))
        pos_eye ='' 
        if max_index==0:
            pos_eye="RIGHT"
            color=[utils.BLACK, utils.GREEN]
        elif max_index==1:
            pos_eye = 'CENTER'
            color = [utils.YELLOW, utils.PINK]       
        elif max_index ==2:
            pos_eye = 'LEFT'
            color = [utils.GRAY, utils.YELLOW]
        elif max_index == 3:
            pos_eye = "UP"
            color = [utils.GRAY, utils.PURPLE]        
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()

    # starting Video loop here.
    while True:

        frame_counter +=1 # frame counter
        ret, frame = camera.read() # getting frame from camera and checking True or False
        if not ret: 
            break # no more frames break
        
        #  resizing frame
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)  # frame size resize
        frame_height, frame_width= frame.shape[:2]  
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)   
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)  
            utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW) ## print on the frame

            if ratio >3.4: 
                CEF_COUNTER +=1              
                utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

            else:
                if CEF_COUNTER>CLOSED_EYES_FRAME:   ## CLOSED_EYES_FRAME = 3  
                    if startBool == True:                    
                        TOTAL_BLINKS +=1
                        CEF_COUNTER =0
                        
                        sending_char = 'b' #blink
                        print('b')
                
                        try:
                            py_serial.write(sending_char.encode())
                        except:
                            pass 
                    
                    n1 = time.time()
                    blinkTimeArray.append(n1)
                    preBlinkIndex = blinkIndex
                    blinkIndex +=1
                    
                    ## blink within 3 seconds
                    if startBool == False:
                        if blinkTimeArray[blinkIndex] - blinkTimeArray[preBlinkIndex] < 3:
                            startBool = True
                            sending_char = 's' # start            
                            try:
                                py_serial.write(sending_char.encode())
                            except:
                                pass 
                            print("blinkTime")
                            utils.colorBackgroundText(frame,  f'Start', FONTS, 1.7, (int(frame_height/2), 600), 2, utils.GRAY,utils.YELLOW, pad_x=6, pad_y=6, )
                            cv.waitKey(150)
                    
                    

            utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            utils.colorBackgroundText(frame,  f'Blink Time: {n1}', FONTS, 0.7, (30,400),2)
            
            
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)

            # Eye Position Detector
            eye_position, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator_left(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            
        
        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        frame =utils.colorBackgroundText(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), textThickness=2)
        cv.imshow('frame', frame)
        key = cv.waitKey(200)

        
        if key==ord('q') or key ==ord('Q'):
            break
        
    cv.destroyAllWindows()
    camera.release()
