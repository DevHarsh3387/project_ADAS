from vehicle import Driver
from controller import Display
import numpy as np
import cv2

driver = Driver()
timestep = int(driver.getBasicTimeStep())

camera = driver.getDevice("driver_camera")
camera.enable(timestep)

display = driver.getDevice("display")

keyboard = driver.getKeyboard()
keyboard.enable(timestep)

joystick = driver.getJoystick()
joystick.enable(timestep)

driver.setCruisingSpeed(20.0)

width = camera.getWidth()
height = camera.getHeight()

MODE_AUTO = "AUTO"
MODE_MANUAL = "MANUAL"
mode = MODE_AUTO

changing_lane = False
lane_change_direction = 0
lane_change_progress = 0
LANE_CHANGE_STEPS = 60
LANE_WIDTH_STEER = 0.25

manual_speed = 0.0
manual_steer = 0.0
MAX_SPEED = 60.0
MAX_REVERSE_SPEED = -10.0
ACCEL_STEP = 0.5
BRAKE_STEP = 0.5
STEER_STEP = 0.03
MAX_STEER = 0.5
STEER_RETURN = 0.02

prev_m_pressed = False


def detect_lanes(image_array):
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    edges = cv2.Canny(mask, 50, 150)

    roi_mask = np.zeros_like(edges)
    roi_vertices = np.array([[
        (0, height),
        (0, int(height * 0.6)),
        (width, int(height * 0.6)),
        (width, height)
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, roi_vertices, 255)
    roi_edges = cv2.bitwise_and(edges, roi_mask)

    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=20,
                             minLineLength=20, maxLineGap=10)

    output = image_array.copy()
    left_x = []
    right_x = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.3:
                left_x.append(x1)
                left_x.append(x2)
            elif slope > 0.3:
                right_x.append(x1)
                right_x.append(x2)

    return output, left_x, right_x


def compute_lane_center_offset(left_x, right_x):
    image_center = width / 2

    if left_x and right_x:
        lane_center = (np.mean(left_x) + np.mean(right_x)) / 2
    elif left_x:
        lane_center = np.mean(left_x) + (width * 0.25)
    elif right_x:
        lane_center = np.mean(right_x) - (width * 0.25)
    else:
        return None

    offset = (lane_center - image_center) / image_center
    return offset


count = 0
while driver.step() != -1:
    count += 1

    pressed_keys = set()
    key = keyboard.getKey()
    while key != -1:
        pressed_keys.add(key)
        key = keyboard.getKey()

    m_pressed = (ord('M') in pressed_keys)
    if m_pressed and not prev_m_pressed:
        mode = MODE_MANUAL if mode == MODE_AUTO else MODE_AUTO
        print(f"Mode switched to: {mode}")
        manual_speed = 0.0
        manual_steer = 0.0
    prev_m_pressed = m_pressed

    image = camera.getImage()

    if image:
        ir_image = display.imageNew(image, Display.BGRA, width, height)
        display.imagePaste(ir_image, 0, 0, False)
        display.imageDelete(ir_image)

    if mode == MODE_MANUAL:
        up_pressed = (keyboard.UP in pressed_keys)
        down_pressed = (keyboard.DOWN in pressed_keys)
        left_pressed = (keyboard.LEFT in pressed_keys)
        right_pressed = (keyboard.RIGHT in pressed_keys)
       
        if up_pressed:
            manual_speed = min(manual_speed + ACCEL_STEP, MAX_SPEED)
        elif down_pressed:
            manual_speed = max(manual_speed - BRAKE_STEP, MAX_REVERSE_SPEED)
        else:
            if manual_speed > 0:
                manual_speed = max(manual_speed - ACCEL_STEP * 0.3, 0)
            elif manual_speed < 0:
                manual_speed = min(manual_speed + ACCEL_STEP * 0.3, 0)

        if left_pressed:
            manual_steer = max(manual_steer - STEER_STEP, -MAX_STEER)
        elif right_pressed:
            manual_steer = min(manual_steer + STEER_STEP, MAX_STEER)
        else:
            if manual_steer > 0:
                manual_steer = max(manual_steer - STEER_RETURN, 0)
            elif manual_steer < 0:
                manual_steer = min(manual_steer + STEER_RETURN, 0)
        if manual_speed > 40:
            speed_limit_factor = 0.4
        elif manual_speed > 25:
            speed_limit_factor = 0.65
        else:
            speed_limit_factor = 1.0
        manual_steer = max(min(manual_steer, MAX_STEER * speed_limit_factor), -MAX_STEER * speed_limit_factor)
        driver.setSteeringAngle(manual_steer)
        driver.setCruisingSpeed(manual_speed)

    else:
        if not changing_lane:
            if key == ord('L'):
                changing_lane = True
                lane_change_direction = 1
                lane_change_progress = 0
            elif key == ord('K'):
                changing_lane = True
                lane_change_direction = -1
                lane_change_progress = 0

        steering_angle = 0.0

        if image:
            img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            bgr_image = cv2.cvtColor(img_array, cv2.COLOR_BGRA2BGR)

            lane_image, left_x, right_x = detect_lanes(bgr_image)
            offset = compute_lane_center_offset(left_x, right_x)

            if changing_lane:
                progress_ratio = lane_change_progress / LANE_CHANGE_STEPS
                steering_angle = lane_change_direction * LANE_WIDTH_STEER * np.sin(np.pi * progress_ratio)
                lane_change_progress += 1
                if lane_change_progress >= LANE_CHANGE_STEPS:
                    changing_lane = False
                    lane_change_progress = 0
            elif offset is not None:
                Kp = 0.4
                steering_angle = Kp * offset
                steering_angle = max(min(steering_angle, 0.5), -0.5)

            driver.setSteeringAngle(steering_angle)
            driver.setCruisingSpeed(20.0)

            if count == 50:
                cv2.imwrite("lane_detection_output.png", lane_image)
                cv2.imwrite("original_camera.png", bgr_image)
                print("Lane detection image saved!")