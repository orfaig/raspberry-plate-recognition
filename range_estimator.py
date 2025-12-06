#sudo lsof | grep /dev/gpio
#sudo kill -9 1518
import RPi.GPIO as GPIO
import time

# --- SETUP (Must run first) ---
GPIO.setmode(GPIO.BOARD) 
GPIO.setwarnings(False) 

AvoidSensorLeft = 21 
AvoidSensorRight = 19 
Avoid_ON = 22 
EchoPin = 18
TrigPin = 16 

GPIO.setup(AvoidSensorLeft, GPIO.IN) 
GPIO.setup(AvoidSensorRight, GPIO.IN) 
GPIO.setup(Avoid_ON, GPIO.OUT) 
GPIO.setup(EchoPin, GPIO.IN) 
GPIO.setup(TrigPin, GPIO.OUT) 

GPIO.output(Avoid_ON, GPIO.HIGH) # Turn on the sensor switch

# --- ULTRASONIC FUNCTIONS (Must be defined after imports and setup) ---

def Distance():
    # Note: I corrected the indentation in your provided code
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW) # This line was cut off in your prompt but is necessary

    t3 = time.time()
    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1
            
    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1

    t2 = time.time()
    # Speed of sound is approx 340 m/s or 34000 cm/s
    # (t2 - t1) is time in seconds. 
    # Distance = (Speed * Time) / 2
    # The result here is in cm because 34000 cm/s was likely used internally by the original source
    return ((t2 - t1) * 34000 / 2) # Use 34000 here to get centimeters directly


def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
        distance = Distance()
        while int(distance) == -1 :
            distance = Distance()
        while (int(distance) >= 500 or int(distance) == 0) :
            distance = Distance()
        ultrasonic.append(distance)
        num = num + 1
        #time.sleep(0.01) # Optional delay
    
    # Average the middle three readings
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3]) / 3
    return distance

# --- MAIN EXECUTION ---
try:
    distance = Distance_test()
    print(f"Average measured distance is: {distance:.2f} cm")

except Exception as e:
    print(f"An error occurred during execution: {e}")

finally:
    GPIO.cleanup()
    print("GPIO pins cleaned up.")

