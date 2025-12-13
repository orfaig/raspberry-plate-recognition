from YB_Pcb_Car_control import YB_Pcb_Car  # import the class directly
import time

# assume the class YB_Pcb_Car is defined above or imported
car = YB_Pcb_Car()

# move forward (both wheels forward)
car.Car_Run(100, 100)   # left speed = 100, right speed = 100
time.sleep(20)           # move forward for 2 seconds

# stop
car.Car_Stop()