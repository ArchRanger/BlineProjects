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
    camera = PiCamera()
    camera.resolution = (640, 360)
    camera.rotation = -90
    rawCapture = PiRGBArray(camera, size=(640, 360))
    time.sleep(0.1)

    x_last = 320
    y_last = 180

    is_forward = True
    reverse_count = 0

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        blackline_image = detect_blackline(image)
        contours_blk_len, contours_blk, hierarchy_blk, x_min, w_min, h_min = find_contours(blackline_image)

        if contours_blk_len > 0:
            reverse_count = 0
            process_image(image, x_last, y_last)
        else:
            reverse_count += 1
            if is_forward:
                GPIO.output(in1, GPIO.HIGH)
                GPIO.output(in2, GPIO.LOW)
                GPIO.output(in3, GPIO.HIGH)
                GPIO.output(in4, GPIO.LOW)
            else:
                GPIO.output(in1, GPIO.LOW)
                GPIO.output(in2, GPIO.HIGH)
                GPIO.output(in3, GPIO.LOW)
                GPIO.output(in4, GPIO.HIGH)
            
            # İleri geri hareket etme süresini ayarlayabilirsiniz, şu anda 0.5 saniye olarak ayarlandı.
            time.sleep(0.5)

            if reverse_count >= 5:
                is_forward = not is_forward
                reverse_count = 0

        rawCapture.truncate(0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

# Diğer fonksiyonlar aynı kaldı...

if __name__ == "__main__":
    main()
