#!/usr/bin/python3
# -*- coding:utf-8 -*-
import RPi.GPIO as GPIO
import time

PIN = 36  # IR pin (BOARD numbering)

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
GPIO.setup(PIN, GPIO.IN, GPIO.PUD_UP)

def get_ir_key(pin=PIN, wait_release=False):
    """
    Decode a single NEC IR frame and return the command byte (0–255).

    Blocking: waits until a valid frame arrives.
    Returns:
        int  - command value (data[2]) if a valid frame was received
        None - if the frame failed integrity check
    """
    ir_repeat_cnt = 0

    # Wait for the line to go low (start of frame)
    while GPIO.input(pin) == 1:
        time.sleep(0.001)

    ir_repeat_cnt = 0
    count = 0

    # 9 ms leading pulse (active low)
    while GPIO.input(pin) == 0 and count < 200:  # 200 * 60us ≈ 12ms
        count += 1
        time.sleep(0.00006)

    # 4.5 ms space (high)
    count = 0
    while GPIO.input(pin) == 1 and count < 80:   # 80 * 60us ≈ 4.8ms
        count += 1
        time.sleep(0.00006)

    idx = 0
    cnt = 0
    data = [0, 0, 0, 0]  # addr, ~addr, cmd, ~cmd

    # 32 bits total
    for i in range(32):
        # 560us low
        count = 0
        while GPIO.input(pin) == 0 and count < 10:
            count += 1
            time.sleep(0.00006)

        # measure length of high to distinguish '0' / '1'
        count = 0
        while GPIO.input(pin) == 1 and count < 40:
            count += 1
            time.sleep(0.00006)

        if count > 9:
            # logic '1'
            data[idx] |= 1 << cnt

        if cnt == 7:
            cnt = 0
            idx += 1
        else:
            cnt += 1

    # NEC integrity check
    if data[0] + data[1] == 0xFF and data[2] + data[3] == 0xFF:
        key = data[2]
        print("Get the key: 0x%02X" % key)
    else:
        print("Invalid frame (checksum failed):",
              [hex(x) for x in data])
        key = None

    # Optional: wait until key is released (line idle for ~110ms)
    if wait_release:
        while True:
            if GPIO.input(pin) == 1:
                if ir_repeat_cnt > 110:
                    ir_repeat_cnt = 0
                    break
                else:
                    time.sleep(0.001)
                    ir_repeat_cnt += 1
            else:
                # still pressed, reset counter
                ir_repeat_cnt = 0

    return key

# Example usage
if __name__ == "__main__":
    print("IR test start...")
    try:
        while True:
            k = get_ir_key(wait_release=True)
            if k is not None:
                print("Command value:", hex(k))
    except KeyboardInterrupt:
        pass
    print("Ending")
    GPIO.cleanup()
