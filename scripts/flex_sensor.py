import serial
import time

# ── CONFIG ─────────────────────────────────────────────────────────
PORT = '/dev/cu.SLAB_USBtoUART' # macOS example — Windows: 'COM3', 'COM4', etc. Linux: '/dev/ttyUSB0'
BAUD_RATE = 115200
# ───────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Give the connection a moment to stabilize
        print(" Connected! Reading flex sensors...\n")
        print("-" * 65)
    except serial.SerialException as e:
        print(f"[ERROR] Could not connect: {e}")
        return

    try:
        while True:
            # 1. Read the raw line and clean off the hidden newline characters
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            # 2. Skip empty lines (sometimes happens on startup)
            if not raw_line:
                continue
            
            # 3. Split the comma-separated string into a python list
            data_parts = raw_line.split(',')
            
            # 4. Make sure we received exactly 11 values before trying to read them
            if len(data_parts) == 11:
                
                # 5. Extract ONLY the first 5 values (Flex sensors)
                thumb = data_parts[0]
                index = data_parts[1]
                middle = data_parts[2]
                ring = data_parts[3]
                pinky = data_parts[4]
                
                # 6. Print them in perfectly aligned columns
                print(f"Thumb: {thumb:<4} | Index: {index:<4} | Middle: {middle:<4} | Ring: {ring:<4} | Pinky: {pinky:<4}")
                
    except KeyboardInterrupt:
        print("\n\nStopped reading. Goodbye!")
    finally:
        # Always safely close the serial port when you quit
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == '__main__':
    main()