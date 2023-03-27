from picamera2 import Picamera2, Preview
import time
picam2 = Picamera2()
time.sleep(5)
picam2.start_and_capture_file("test_cam.jpg")
