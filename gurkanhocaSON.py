from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
GPIO.cleanup(False)
# Motor pinlerini tanımla
in1 = 4
in2 = 17
in3 = 27
in4 = 22
en1 = 23
en2 = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(en1, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
p1 = GPIO.PWM(en1, 100)
p2 = GPIO.PWM(en2, 100)
p1.start(50)
p2.start(50)
GPIO.output(in1, GPIO.LOW)
GPIO.output(in2, GPIO.LOW)
GPIO.output(in3, GPIO.LOW)
GPIO.output(in4, GPIO.LOW)

def main():
    # GPIO.setwarnings(True)  # GPIO uyarılarını kapat
#     GPIO.setmode(GPIO.BOARD)
#     GPIO.cleanup()  # Önceki GPIO ayarlarını temizle



    # Motor pinlerini çıkış olarak ayarla
#     GPIO.setup(motor1_pwm0_pin, GPIO.OUT)
#     GPIO.setup(motor1_pwm1_pin, GPIO.OUT)
#     GPIO.setup(motor2_pwm0_pin, GPIO.OUT)
#     GPIO.setup(motor2_pwm1_pin, GPIO.OUT)

    # Motorları durdur
#     stop_motors()

    camera = PiCamera()
    camera.resolution = (640, 360)
    camera.rotation = -90
    rawCapture = PiRGBArray(camera, size=(640, 360))
    time.sleep(0.1)

    x_last = 320
    y_last = 180

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        process_image(image, x_last, y_last)
        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def process_image(image, x_last, y_last):
    blackline_image = detect_blackline(image)

    contours_blk_len, contours_blk, hierarchy_blk, x_min, w_min, h_min = find_contours(blackline_image)

    if contours_blk_len > 0:
        blackbox, x_last, y_last = calculate_blackbox(contours_blk, contours_blk_len, x_last, y_last)
        draw_blackbox(image, blackbox)
        display_angle_and_error(image, blackbox, x_min, w_min, h_min)
        display_line(image, blackbox)

        # Çizgiye göre motorları kontrol et
        control_motors(image, blackbox)

    cv2.imshow("Original with line", image)


def detect_blackline(image):
    blackline = cv2.inRange(image, (0, 0, 0), (65, 65, 65))
    kernel = np.ones((3, 3), np.uint8)
    blackline = cv2.erode(blackline, kernel, iterations=5)
    blackline = cv2.dilate(blackline, kernel, iterations=9)
    return blackline


def find_contours(blackline_image):
    contours_blk, hierarchy_blk = cv2.findContours(blackline_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_blk_len = len(contours_blk)
    if contours_blk_len > 0:
        blackbox = cv2.minAreaRect(contours_blk[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        return contours_blk_len, contours_blk, hierarchy_blk, x_min, w_min, h_min
    return contours_blk_len, [], [], 0, 0, 0


def calculate_blackbox(contours_blk, contours_blk_len, x_last, y_last):
    if contours_blk_len == 1:
        blackbox = cv2.minAreaRect(contours_blk[0])
    else:
        blackbox, x_last, y_last = handle_multiple_contours(contours_blk, contours_blk_len, x_last, y_last)
    return blackbox, x_last, y_last


def handle_multiple_contours(contours_blk, contours_blk_len, x_last, y_last):
    canditates = []
    off_bottom = 0
    for con_num in range(contours_blk_len):
        blackbox = cv2.minAreaRect(contours_blk[con_num])
        (x_min, y_min), _, _ = blackbox
        if y_min > 358:
            off_bottom += 1
        canditates.append((y_min, con_num, x_min, y_min))
    canditates = sorted(canditates)
    if off_bottom > 1:
        canditates_off_bottom = []
        for con_num in range((contours_blk_len - off_bottom), contours_blk_len):
            (_, con_highest, x_min, y_min) = canditates[con_num]
            total_distance = (abs(x_min - x_last) ** 2 + abs(y_min - y_last) ** 2) ** 0.5
            canditates_off_bottom.append((total_distance, con_highest))
        canditates_off_bottom = sorted(canditates_off_bottom)
        (_, con_highest) = canditates_off_bottom[0]
        blackbox = cv2.minAreaRect(contours_blk[con_highest])
    else:
        (_, con_highest, x_min, y_min) = canditates[contours_blk_len - 1]
        blackbox = cv2.minAreaRect(contours_blk[con_highest])
    (x_min, y_min), _, _ = blackbox
    x_last = x_min
    y_last = y_min
    return blackbox, x_last, y_last


def draw_blackbox(image, blackbox):
    box = cv2.boxPoints(blackbox)
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 3)


def display_angle_and_error(image, blackbox, x_min, w_min, h_min):
    (_, _), (_, _), ang = blackbox
    ang = handle_angle(ang, w_min, h_min)
    error = int(x_min - 320)
    cv2.putText(image, str(ang), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image, str(error), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def handle_angle(ang, w_min, h_min):
    if ang < -45:
        ang = 90 + ang
    if w_min < h_min and ang > 0:
        ang = (90 - ang) * -1
    if w_min > h_min and ang < 0:
        ang = 90 + ang
    return int(ang)


def display_line(image, blackbox):
    (x_min, _), (_, _), _ = blackbox
    cv2.line(image, (int(x_min), 200), (int(x_min), 250), (255, 0, 0), 3)


def control_motors(image, blackbox):
    (x_min, _), (_, _), _ = blackbox
    error = x_min - 320
    # Hata miktarına göre motor hızını ayarla
#     motor_speed = 50  # Örnek bir hız, ihtiyaca göre değiştirilebilir
#     motor1_pwm0_speed = motor_speed
#     motor1_pwm1_speed = motor_speed
#     motor2_pwm0_speed = motor_speed
#     motor2_pwm1_speed = motor_speed

    
    
    if abs(error) < 90:  # Eğer hat hata miktarı belirli bir eşiğin altındaysa doğru yönde ilerle
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
    elif error > 90:  # Eğer hata pozitif ise sağa dön
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
    elif error < -90:  # Eğer hata negatif ise sola dön
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)

    else:
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.LOW)


def stop_motors():
    GPIO.output(in1, GPIO.LOW)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.LOW)


if __name__ == "__main__":
    main()

