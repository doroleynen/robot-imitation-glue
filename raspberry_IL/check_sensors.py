"""
Quick sensor diagnostic — checks all data streams without connecting to the robot arm.

Usage:
    python raspberry_IL/check_sensors.py
    python raspberry_IL/check_sensors.py --no-anyskin   # skip AnySkin if not connected
"""

import argparse
import threading
import time

import serial
from anyskin import AnySkinBase

from raspberry_IL.uR3station.raspberry_trial_utils import parse_loadcell_line, parse_raspberry_line

RASPBERRY_PORT = "/dev/ttyACM2"
LOADCELL_PORT  = "/dev/ttyACM3"
ANYSKIN_PORT   = "/dev/ttyACM1"
BAUD_RATE      = 115200
REFRESH_HZ     = 5   # print update rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raspberry-port", default=RASPBERRY_PORT)
    parser.add_argument("--loadcell-port",  default=LOADCELL_PORT)
    parser.add_argument("--anyskin-port",   default=ANYSKIN_PORT)
    parser.add_argument("--no-anyskin", action="store_true")
    args = parser.parse_args()

    state = {
        "raspberry": [None] * 8,
        "raspberry_count": 0,
        "loadcell": None,
        "loadcell_count": 0,
        "anyskin": None,
        "anyskin_count": 0,
    }
    lock = threading.Lock()
    stop = threading.Event()

    def raspberry_reader():
        try:
            ser = serial.Serial(args.raspberry_port, BAUD_RATE, timeout=1)
            print(f"[raspberry] opened {args.raspberry_port}")
        except Exception as e:
            print(f"[raspberry] FAILED to open {args.raspberry_port}: {e}")
            return
        while not stop.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
                vals = parse_raspberry_line(line)
                if vals is None:
                    continue
                with lock:
                    state["raspberry"] = [vals.get(f"S{i}") for i in range(8)]
                    state["raspberry_count"] += 1
            except Exception:
                pass

    def loadcell_reader():
        try:
            ser = serial.Serial(args.loadcell_port, BAUD_RATE, timeout=1)
            print(f"[loadcell]  opened {args.loadcell_port}")
        except Exception as e:
            print(f"[loadcell]  FAILED to open {args.loadcell_port}: {e}")
            return
        while not stop.is_set():
            try:
                line = ser.readline().decode(errors="ignore").strip()
                force = parse_loadcell_line(line)
                if force is None:
                    continue
                with lock:
                    state["loadcell"] = force
                    state["loadcell_count"] += 1
            except Exception:
                pass

    def anyskin_reader():
        try:
            sensor = AnySkinBase(num_mags=5, port=args.anyskin_port,
                                 baudrate=BAUD_RATE, burst_mode=True, temp_filtered=True)
            print(f"[anyskin]   opened {args.anyskin_port}")
        except Exception as e:
            print(f"[anyskin]   FAILED to open {args.anyskin_port}: {e}")
            return
        while not stop.is_set():
            try:
                _, sample = sensor.get_sample()
                with lock:
                    state["anyskin"] = list(sample)
                    state["anyskin_count"] += 1
            except Exception:
                pass

    threads = [
        threading.Thread(target=raspberry_reader, daemon=True),
        threading.Thread(target=loadcell_reader, daemon=True),
    ]
    if not args.no_anyskin:
        threads.append(threading.Thread(target=anyskin_reader, daemon=True))

    for t in threads:
        t.start()

    time.sleep(1.0)  # let threads open ports
    print("\nStreaming — press Ctrl+C to stop\n")

    try:
        while True:
            time.sleep(1.0 / REFRESH_HZ)
            with lock:
                rasp  = state["raspberry"]
                lc    = state["loadcell"]
                skin  = state["anyskin"]
                rc    = state["raspberry_count"]
                lcc   = state["loadcell_count"]
                skc   = state["anyskin_count"]

            rasp_str = "  ".join(
                f"S{i}={v:>7.1f}" if v is not None else f"S{i}=   N/A "
                for i, v in enumerate(rasp)
            )
            lc_str  = f"{lc:>8.3f} N" if lc is not None else "     N/A"

            if skin is not None:
                # show active mags 0 and 4 (xyz each)
                m0 = skin[0:3]
                m4 = skin[12:15]
                skin_str = (f"m0=({m0[0]:6.1f},{m0[1]:6.1f},{m0[2]:6.1f})  "
                            f"m4=({m4[0]:6.1f},{m4[1]:6.1f},{m4[2]:6.1f})")
            else:
                skin_str = "N/A"

            print(
                f"\r[rasp #{rc:>5}] {rasp_str}  "
                f"[load #{lcc:>5}] {lc_str}  "
                f"[skin #{skc:>5}] {skin_str}",
                end="", flush=True
            )

    except KeyboardInterrupt:
        print("\n\nStopped.")
        stop.set()

        with lock:
            rc  = state["raspberry_count"]
            lcc = state["loadcell_count"]
            skc = state["anyskin_count"]

        print(f"\nTotal samples received:")
        print(f"  Raspberry : {rc}")
        print(f"  Loadcell  : {lcc}")
        print(f"  AnySkin   : {skc}")
        if rc == 0:   print("  WARNING: no raspberry data received — check port/cable")
        if lcc == 0:  print("  WARNING: no loadcell data received — check port/cable")
        if skc == 0 and not args.no_anyskin:
            print("  WARNING: no AnySkin data received — check port/cable")


if __name__ == "__main__":
    main()
