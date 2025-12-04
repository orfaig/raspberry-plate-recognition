# forward_test_gpiozero.py
from time import sleep
from gpiozero import Motor

# CHANGE THESE PINS to your wiring:
LEFT_IN1, LEFT_IN2, LEFT_EN  = 17, 27, 22
RIGHT_IN1, RIGHT_IN2, RIGHT_EN = 23, 24, 25

left  = Motor(forward=LEFT_IN1,  backward=LEFT_IN2,  enable=LEFT_EN,  pwm=True)
right = Motor(forward=RIGHT_IN1, backward=RIGHT_IN2, enable=RIGHT_EN, pwm=True)

speed = 0.6      # 0.0 .. 1.0
duration = 2.0   # seconds

try:
    print("Driving forward...", flush=True)
    left.forward(speed)
    right.forward(speed)
    sleep(duration)
finally:
    print("Stopping.", flush=True)
    left.stop()
    right.stop()
