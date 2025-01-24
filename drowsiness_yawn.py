from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from imutils.video import VideoStream
import imutils
import dlib
import cv2
import numpy as np
from threading import Thread
import os
from scipy.spatial import distance as dist
from imutils import face_utils
from playsound import playsound
import time

# Global alarm flags
alarm_status = False
alarm_status2 = False
saying = False
DROWSINESS_COUNTER = 0  # Separate counter for drowsiness detection
BLINK_COUNTER = 0  # Separate counter for blinks within 12-second interval
START_TIME = time.time()  # Start time of the current 12-second window
BLINK_COUNT = 0  # Count blinks within 12-second window
BLINK_THRESHOLD = 5  # Number of blinks for warning
TIME_WINDOW = 12  # Time window in seconds
EAR_THRESHOLD = 0.25  # EAR threshold to detect a blink
CONSECUTIVE_FRAMES = 3  # Blink should last for at least 3 consecutive frames
DROWSINESS_THRESHOLD = 30  # Consecutive frames for drowsiness detection

# Drowsiness/Yawn Detection Logic
def alarm(msg, sound_file=None):
    global alarm_status
    global alarm_status2
    global saying

    if sound_file:
        playsound(sound_file)

    while alarm_status:
        s = 'espeak "' + msg + '"'
        os.system(s)

    if alarm_status2:
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    try:
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)
    except IndexError:
        return None

def lip_distance(shape):
    try:
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        return distance
    except IndexError:
        return None

class DrowsinessApp(App):
    def build(self):
        self.detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.vs = VideoStream(src=0).start()

        # GUI Layout
        self.layout = BoxLayout(orientation='vertical')
        self.image_widget = Image(size_hint=(1, 0.8))
        self.status_label = Label(text="Drowsiness Detection System", size_hint=(1, 0.1))
        self.alert_label = Label(text="Status: Normal", size_hint=(1, 0.1))

        self.layout.add_widget(self.image_widget)
        self.layout.add_widget(self.status_label)
        self.layout.add_widget(self.alert_label)

        Clock.schedule_interval(self.update, 1.0/30.0)  # Call update 30 times per second
        return self.layout

    def update(self, dt):
        global DROWSINESS_COUNTER, alarm_status, alarm_status2, BLINK_COUNT, BLINK_COUNTER, START_TIME

        frame = self.vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if 12 seconds have passed
        current_time = time.time()
        elapsed_time = current_time - START_TIME

        if len(rects) > 0:
            for (x, y, w, h) in rects:
                rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                ear_data = final_ear(shape)
                distance = lip_distance(shape)

                if ear_data and distance:
                    ear = ear_data[0]

                    # Blink Detection Logic
                    if ear < EAR_THRESHOLD:
                        BLINK_COUNTER += 1
                        if BLINK_COUNTER >= CONSECUTIVE_FRAMES:  # Ensure blink lasts 3 frames
                            BLINK_COUNT += 1
                            BLINK_COUNTER = 0  # Reset blink counter
                    else:
                        BLINK_COUNTER = 0

                    # Drowsiness Detection Logic
                    if ear < 0.3:  # EAR threshold for drowsiness
                        DROWSINESS_COUNTER += 1
                        if DROWSINESS_COUNTER >= DROWSINESS_THRESHOLD:  # Check for 30 consecutive frames
                            self.alert_label.text = "Status: Drowsiness Alert!"
                            if not alarm_status:
                                alarm_status = True
                                Thread(target=alarm, args=("Wake up, please!", "drowsy-alert.mp3")).start()
                    else:
                        DROWSINESS_COUNTER = 0  # Reset counter if no drowsiness is detected
                        alarm_status = False  # Reset alarm status

                    # Yawn Alert
                    if distance > 20:
                        self.alert_label.text = "Status: Yawning Alert!"
                        if not alarm_status2 and not saying:
                            alarm_status2 = True
                            Thread(target=alarm, args=("Take a break!", "yawn-alert.wav")).start()
                    else:
                        alarm_status2 = False

                    # Update EAR and Yawn Distance on screen
                    self.status_label.text = f"EAR: {ear:.2f}, Yawn: {distance:.2f}"

                # Convert Frame for Kivy
                buf = cv2.flip(frame, 0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.image_widget.texture = texture
        else:
            self.alert_label.text = "Status: No face detected"

        # Check if 12 seconds have passed to evaluate blink count
        if elapsed_time >= TIME_WINDOW:
            if BLINK_COUNT >= BLINK_THRESHOLD:
                self.alert_label.text = "Status: Frequent Blinking Warning!"
                Thread(target=alarm, args=("Frequent blinking detected!", "blink-warning.mp3")).start()
            else:
                self.alert_label.text = "Status: Normal Blinking"

            # Reset blink count and start time for the next 12-second window
            BLINK_COUNT = 0
            START_TIME = current_time

    def on_stop(self):
        self.vs.stop()

if __name__ == '__main__':
    DrowsinessApp().run()

