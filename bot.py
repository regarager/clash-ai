from time import time, sleep
from adb_pywrapper.adb_device import AdbDevice

from positions import *


class Bot:
    def __init__(self, device: str) -> None:
        self.device: str = device
        self.adb: AdbDevice = AdbDevice(device)

    def is_offline(self):
        return AdbDevice.get_device_status(self.device) == "offline"

    def shell(self, cmd: str):
        _ = self.adb.shell(cmd)

    def tap(self, p: tuple[int, int]):
        self.shell(f"input mouse tap {p[0]} {p[1]}")

    def battle(self):
        self.tap(BATTLE)

    def play_card(self, i: int, x: float, y: float):
        self.tap(CARDS[i])
        sleep(0.5)
        self.tap(ALLY_CORNERS.pt(x, y))

    def screenshot(self) -> str:
        filename = f"screenshots/{int(time())}.png"
        print(f"Screenshotted to {filename}")
        self.shell("settings put system pointer_location 0")
        self.shell(f"screencap -p > {filename}")
        self.shell("settings put system pointer_location 1")

        return filename


