from time import time
from typing import Tuple

from adb_pywrapper.adb_device import AdbDevice

from positions import BATTLE

devices = AdbDevice.list_devices()

device = AdbDevice(devices[0])
print("Found devices", devices)
print("Using first device:", device)

print("Status:", device.get_device_status(devices[0]))


def shell(cmd: str):
    return device.shell(cmd)


def tap(p: Tuple[int, int]):
    shell(f"input mouse tap {p[0]} {p[1]}")


def screenshot():
    filename = f"screenshots/{int(time())}.png"
    print(f"Screenshotted to {filename}")
    shell(f"screencap -p > {filename}")


def resize(w: int, h: int):
    shell(f"wm size {w}x{h}")


shell("settings put system pointer_location 1")
size = shell("wm size")
# resize(640, 640)

tap(BATTLE)
