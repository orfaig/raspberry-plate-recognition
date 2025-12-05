#!/usr/bin/python3
# -*- coding:utf-8 -*-
import RPi.GPIO as GPIO
import time

PIN = 36;   #Define IR pin

#Set the GPIO port to BIARD encoding mode
GPIO.setmode(GPIO.BOARD)

#Ignore the warning message
GPIO.setwarnings(False)
ir_repeat_cnt = 0

#The pin of the red external device needs to be set to input pull-up
GPIO.setup(PIN,GPIO.IN,GPIO.PUD_UP)
print("IR test start...")  #Initially print "IR test start".

try:
    print("start")
    while True:
        if GPIO.input(PIN) == 0:   #The signal emitted by the infrared remote control is detected
            ir_repeat_cnt = 0;
            count = 0
            while GPIO.input(PIN) == 0 and count < 200:   #Judge the boot code of 9ms high level pulse
                count += 1
                time.sleep(0.00006)

            count = 0
            while GPIO.input(PIN) == 1 and count < 80:   #Judge the boot code of 4.5ms low-level pulse
                count += 1
                time.sleep(0.00006)

            idx = 0
            cnt = 0
            data = [0,0,0,0]   #Define data used to store the address code, address inversion, signal code, and signal inversion of infrared signals
            for i in range(0,32):   #data[0],data[1],data[2],data[3] In total, 8bit*4=32
                count = 0
                while GPIO.input(PIN) == 0 and count < 10:   #Start decoding, used to filter the first 560us pulse of logic 0 and logic 1
                    count += 1
                    time.sleep(0.00006)

                count = 0
                while GPIO.input(PIN) == 1 and count < 40:   #After the 560us high-level pulse, check the remaining low-level pulse time length to determine whether it is logic 0 or logic 1
                    '''
                    Description:
                    According to the infrared NCE agreement:
                    The period of logic 1 is 2.25ms, and the pulse time is 0.56ms.Total period-pulse time = the value we set, the set value is slightly larger than the actual value.

                    The logic 0 period is 1.12 and the time is 0.56ms. Total period-pulse time = the value we set, the set value is slightly larger than the actual value.
                    
                    '''
                    count += 1
                    time.sleep(0.00006)

                if count > 9:    
                    #This code is used to determine whether the currently received signal is logic 1 or logic 0.
                    #If count>9, it proves that the duration of the current low-level signal is greater than 560 (9*60=540us), which is logic 1.
                    #For example: when count=10, the low-level signal is 10*60=600us (greater than 560us)，it is logic 1.
                    
                    data[idx] |= 1<<cnt   
                if cnt == 7:   #When cnt=7, one byte is full, and the next byte is ready to be stored.
                    cnt = 0
                    idx += 1
                else:
                    cnt += 1  
                    
            if data[0]+data[1] == 0xFF and data[2]+data[3] == 0xFF:  #It is judged that the correct infrared remote control code value is received.
                print("Get the key: 0x%02x" %data[2])   #Print the command code obtained
        else:
            if ir_repeat_cnt > 110: #Judge whether the infrared remote control button is released, because the repetition cycle time is 110ms, so here it should be set to 110*0.001.
                ir_repeat_cnt = 0
            else:
                time.sleep(0.001)
                ir_repeat_cnt += 1
except KeyboardInterrupt:
    pass
print("Ending")
GPIO.cleanup()