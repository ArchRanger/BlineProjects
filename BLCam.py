from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from flask import Flask, Response

# Flask uygulaması
app = Flask(__name__)

GPIO.setwarnings(False)  # GPIO uyarılarını devre dışı bırak
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

# PID parametreleri
Kp = 0.7
Ki = 0.0
Kd = 0.1

# PID değişkenleri
last_error = 0
integral = 0

camera = PiCamera()
camera.resolution = (640, 360)
camera.rotation = -90
rawCapture = PiRGBArray(camera, size=(640, 360))
time.sleep(0.1)

x_last = 320
y_last = 180

def generate_frames():
    global x_last, y_last
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        process_image(image, x_last, y_last)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        rawCapture.truncate(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    else:
        # Eğer çizgi algılanmazsa, geriye doğru git
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        time.sleep(0.5)  # Geri hareket süresi, ihtiyaca göre ayarlayabilirsiniz

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
    global last_error, integral

    (x_min, _), (_, _), _ = blackbox
    error = x_min - 320

    # PID hesaplama
    P = Kp * error
    I = Ki * integral
    D = Kd * (error - last_error)

    pid_output = P + I + D

    # PID çıkışını sınırlama
    pid_output = max(min(pid_output, 100), -100)

    # PID çıkışını motor hızına dönüştürme
    motor_speed = abs(pid_output)

    if error < -90:  # Eğer hata negatif ise sola dön
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.HIGH)
        GPIO.output(in4, GPIO.LOW)
        p1.ChangeDutyCycle(motor_speed)
        p2.ChangeDutyCycle(motor_speed)
    elif error > 90:  # Eğer hata pozitif ise sağa dön
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        p1.ChangeDutyCycle(motor_speed)
        p2.ChangeDutyCycle(motor_speed)
    else:  # Eğer hata belirtilen aralıkta ise ileri git
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
        GPIO.output(in3, GPIO.LOW)
        GPIO.output(in4, GPIO.HIGH)
        p1.ChangeDutyCycle(50)  # Sabit bir hız
        p2.ChangeDutyCycle(50)  # Sabit bir hız

    # PID değişkenlerini güncelleme
    last_error = error
    integral += error

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
