from time import time, sleep
from adb_pywrapper.adb_device import AdbDevice
from .positions import BATTLE, CARDS, ALLY_CORNERS
from .minicap import Minicap


class Bot:
    def __init__(self, device: str) -> None:
        self.device: str = device
        self.adb: AdbDevice = AdbDevice(device)
        self.minicap: Minicap = Minicap(device)

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

    def check_minicap(self):
        """Runs the diagnostic check for minicap installation."""
        self.minicap.check_installation()

    def screenshot(self) -> str:
        """
        Takes a screenshot using minicap.
        Raises RuntimeError if minicap is not functional.
        """
        self.shell("settings put system pointer_location 0")
        filename = self.minicap.screenshot()
        self.shell("settings put system pointer_location 1")
        print(f"Screenshotted to {filename}")

        return filename

    def get_screen_size(self) -> tuple[int, int]:
        """
        Gets the current screen size of the device using 'wm size' adb command.
        Returns:
            tuple[int, int]: A tuple containing (width, height) of the screen.
        """
        output = self.adb.shell("wm size")
        # Expected output format: "Physical size: WxH"
        # Example: "Physical size: 1432x1736"
        parts = output.stdout.strip().split(": ")
        if len(parts) == 2:
            dimensions = parts[1].split("x")
            if len(dimensions) == 2:
                width = int(dimensions[0])
                height = int(dimensions[1])
                return width, height
        print(
            f"Warning: Could not parse screen size from adb output: {output}. Defaulting to 1920x1080."
        )
        return 1920, 1080
