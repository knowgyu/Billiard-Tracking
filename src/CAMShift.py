# CAM Shift 알고리즘을 적용한 코드입니다.
import cv2
import numpy as np

point_list = []
pts_cnt = 0 

def mouse_callback(event, x, y, flags, params):
    global point_list, pts_cnt, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" %(x,y))
        point_list.append((x,y))
        pts_cnt += 1

        print(point_list)
        cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow('original',frame)

cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)

cap = cv2.VideoCapture('video/conv/longback.avi')
font,LT =cv2.FONT_HERSHEY_SIMPLEX,cv2.LINE_AA

while True:
    _, frame = cap.read()
    cv2.putText(frame,"Click Corners and Press q",(30,30),font,1,(255,255,255),0,LT)
    cv2.imshow('original',frame)
    height, width = frame.shape[:2]

    k = cv2.waitKey(1000) & 0xFF
    if pts_cnt == 4:
        break


### 순서 상관없이!!
pts = np.array(point_list,dtype=np.float32)
if pts_cnt ==4:
    print(pts)
    # 좌표 4개 중 상하좌우 찾기
    sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

    topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
    bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 우하단 좌표
    topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

    # 변환 전 4개 좌표 
    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
    height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])

M = cv2.getPerspectiveTransform(pts1,pts2)

while True:
    _, frame = cap.read()
    tf = cv2.warpPerspective(frame, M, (round(width),round(height)))

    cv2.imshow('tf', tf)
    k = cv2.waitKey(10000)
    if k == ord('q'):
        break

trackWindow = cv2.selectROI('tf',tf)
x,y,w,h = trackWindow
roi = frame[y:y+h, x:x+w]
roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist=cv2.calcHist([roi],[0],None,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
termination=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)

while True:
    ret, frame=cap.read()
    if not ret:
        break
    if trackWindow is not None:
        frame = cv2.warpPerspective(frame, M, (round(width),round(height)))
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, trackWindow=cv2.CamShift(dst,trackWindow,termination)
        cv2.ellipse(frame, ret, (0,0,255),2)
    cv2.imshow('frame',frame)
    k=cv2.waitKey(60) &0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()