import time
from remote_fucntion import get_ir_key

print("Press buttons on the remote. Valid decoded keys will be printed as hex.")
print("If you only see 'Invalid frame' messages and nothing else, IR decoding is failing.")

try:
    while True:
        key = get_ir_key()
        if key is not None:
            print(f"Decoded key: 0x{key:02X}")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nExiting IR test.")
