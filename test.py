import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import time
import RPi.GPIO as GPIO
import tensorflow as tf

class stateCtrl(object):
    '''Motor Control Module'''

    def __init__(self):
        # Pin Definitions
        self.motor1_pin1 = 11
        self.motor1_pin2 = 12
        self.motor2_pin1 = 13
        self.motor2_pin2 = 15
        self.enable_pin1 = 33
        self.enable_pin2 = 35
        self.setup()

    def setup(self):
        '''Pin initialization'''
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)

        # Set up the GPIO pins as outputs
        GPIO.setup(self.motor1_pin1, GPIO.OUT)
        GPIO.setup(self.motor1_pin2, GPIO.OUT)
        GPIO.setup(self.motor2_pin1, GPIO.OUT)
        GPIO.setup(self.motor2_pin2, GPIO.OUT)
        GPIO.setup(self.enable_pin1, GPIO.OUT)
        GPIO.setup(self.enable_pin2, GPIO.OUT)

        # Create PWM objects for motor control
        self.pwm1 = GPIO.PWM(self.enable_pin1, 100)
        self.pwm2 = GPIO.PWM(self.enable_pin2, 100)

        # Start PWM
        self.pwm1.start(0)
        self.pwm2.start(0)

    def move_forward(self):
        GPIO.output(self.motor1_pin1, GPIO.HIGH)
        GPIO.output(self.motor1_pin2, GPIO.LOW)
        GPIO.output(self.motor2_pin1, GPIO.HIGH)
        GPIO.output(self.motor2_pin2, GPIO.LOW)
        self.pwm1.ChangeDutyCycle(70)
        self.pwm2.ChangeDutyCycle(70)

    def move_backward(self):
        GPIO.output(self.motor1_pin1, GPIO.LOW)
        GPIO.output(self.motor1_pin2, GPIO.HIGH)
        GPIO.output(self.motor2_pin1, GPIO.LOW)
        GPIO.output(self.motor2_pin2, GPIO.HIGH)
        self.pwm1.ChangeDutyCycle(70)
        self.pwm2.ChangeDutyCycle(70)

    def turn_left(self):
        GPIO.output(self.motor1_pin1, GPIO.LOW)
        GPIO.output(self.motor1_pin2, GPIO.HIGH)
        GPIO.output(self.motor2_pin1, GPIO.HIGH)
        GPIO.output(self.motor2_pin2, GPIO.LOW)
        self.pwm1.ChangeDutyCycle(70)
        self.pwm2.ChangeDutyCycle(70)

    def turn_right(self):
        GPIO.output(self.motor1_pin1, GPIO.HIGH)
        GPIO.output(self.motor1_pin2, GPIO.LOW)
        GPIO.output(self.motor2_pin1, GPIO.LOW)
        GPIO.output(self.motor2_pin2, GPIO.HIGH)
        self.pwm1.ChangeDutyCycle(70)
        self.pwm2.ChangeDutyCycle(70)

    def stop(self):
        GPIO.output(self.motor1_pin1, GPIO.LOW)
        GPIO.output(self.motor1_pin2, GPIO.LOW)
        GPIO.output(self.motor2_pin1, GPIO.LOW)
        GPIO.output(self.motor2_pin2, GPIO.LOW)
        self.pwm1.ChangeDutyCycle(0)
        self.pwm2.ChangeDutyCycle(0)

# Create an instance of the stateCtrl class
# car = stateCtrl()

# Open the video capture device (webcam)

def process_emotion():
    car = stateCtrl()
    # load model
    model = load_model("best_model.h5")
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (224, 224))
            img_pixels = tf.keras.utils.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            # find max indexed array
            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if predicted_emotion == 'angry' or predicted_emotion == 'sad':
                car.move_forward()  # Move the car forward
                time.sleep(10)
                car.stop()
                # # Release the video capture device and close all windows
                print('ANGRYYYYYYYYYYYYYYY')
                break

        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows

# Start the facial emotion analysis
process_emotion()

