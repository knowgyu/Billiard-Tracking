import cv2
import numpy as np
from math import pi
import argparse

# ===================== 공 경로 그리는 함수 ========================= #
def draw_ball_location(img_color, locations, t_color):
    for i in range(len(locations) - 1):
        if locations[0] is None or locations[1] is None:
            continue
        cv2.line(img_color, tuple(locations[i]), tuple(locations[i + 1]), t_color, 2)
    return img_color

# ===================== 마우스 이벤트 처리 ========================= #
def handle_mouse_events(event, x, y, flags, params):
    global point_list, pts_cnt, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")
        point_list.append((x, y))
        pts_cnt += 1
        cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow('original', frame)

# ===================== Perspective 변환 행렬 계산 ========================= #
def get_perspective_transform(points):
    sm = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    topLeft = points[np.argmin(sm)]
    bottomRight = points[np.argmax(sm)]
    topRight = points[np.argmin(diff)]
    bottomLeft = points[np.argmax(diff)]

    pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    width, height = max(w1, w2), max(h1, h2)
    pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    return cv2.getPerspectiveTransform(pts1, pts2), (int(width), int(height))

# ===================== 프레임 처리 및 공 탐지 ========================= #
def process_frame(frame, M, width, height, list_whiteball_location, list_yellowball_location):
    frame = cv2.warpPerspective(frame, M, (width, height))
    img_blur = cv2.GaussianBlur(frame, (7, 7), 0)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        mmt = cv2.moments(cnt)
        if mmt['m00'] == 0: continue
        area = mmt['m00']
        if area < 450 or area > 700: continue

        cx, cy = int(mmt['m10'] / area), int(mmt['m01'] / area)
        perimeter = cv2.arcLength(cnt, True)
        circular = 4 * pi * area / (perimeter * perimeter)
        color = frame[cy, cx]

        if color.min() > 225 and circular > 0.85:  # 흰공
            list_whiteball_location.append((cx, cy))
            draw_ball_location(frame, list_whiteball_location, (255, 255, 255))
        elif color.min() < 70 and circular > 0.72:  # 노란공
            list_yellowball_location.append((cx, cy))
            draw_ball_location(frame, list_yellowball_location, (0, 255, 255))

    return frame, img_bin

# ===================== 메인 함수 ========================= #
def main(video_path):
    global point_list, pts_cnt, frame
    point_list, pts_cnt = [], 0
    list_whiteball_location, list_yellowball_location = [], []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    cv2.namedWindow('original')
    cv2.setMouseCallback('original', handle_mouse_events)

    while True:
        _, frame = cap.read()
        if frame is None: break
        cv2.putText(frame, "Click 4 Corners", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 0, cv2.LINE_AA)
        cv2.imshow('original', frame)
        if pts_cnt == 4: break
        cv2.waitKey(20)

    M, (width, height) = get_perspective_transform(np.array(point_list, dtype=np.float32))

    while True:
        ret, frame = cap.read()
        if not ret: break
        processed_frame, binary_frame = process_frame(frame, M, width, height, list_whiteball_location, list_yellowball_location)
        cv2.imshow('binary', binary_frame)
        cv2.imshow('result', processed_frame)
        if cv2.waitKey(20) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

# ===================== argparse를 사용한 실행 ========================= #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track balls in a video file.")
    parser.add_argument('video_path', type=str, help="Path to the video file.")
    args = parser.parse_args()
    main(args.video_path)