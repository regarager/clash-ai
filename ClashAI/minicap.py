import os
import re
from time import time
from typing import Optional, Tuple

from adb_pywrapper.adb_device import AdbDevice

from .logger import *


class Minicap:
    """
    A class to handle screenshots using minicap for faster screen capture.
    Reference: https://github.com/DeviceFarmer/minicap
    """

    def __init__(self, device_id: str, work_dir: str = "/data/local/tmp"):
        """
        Initialize the Minicap handler.
        :param device_id: The ADB device identifier.
        :param work_dir: The directory on the device where minicap files are located.
        """
        self.device_id = device_id
        self.adb = AdbDevice(device_id)
        self.work_dir = work_dir
        self.bin_path = f"{work_dir}/minicap"
        self.so_path = f"{work_dir}/minicap.so"
        self._screen_size: Optional[Tuple[int, int]] = None
        self._rotation: Optional[int] = None
        self._is_installed: Optional[bool] = None

    def get_screen_size(self) -> Tuple[int, int]:
        """Gets the screen size from the device using 'wm size'."""
        if self._screen_size is None:
            output = self.adb.shell("wm size").stdout
            match = re.search(r"Physical size: (\d+)x(\d+)", output)
            if match:
                self._screen_size = (int(match.group(1)), int(match.group(2)))
            else:
                # Fallback if parsing fails
                self._screen_size = (1080, 1920)
        return self._screen_size

    def get_rotation(self) -> int:
        """Gets the current screen rotation (0, 90, 180, 270) from dumpsys."""
        output = self.adb.shell(
            "dumpsys window | grep -E 'mCurrentRotation|mRotation'"
        ).stdout
        match = re.search(r"m(?:Current)?Rotation=(\d+)", output)
        if match:
            # rotation values are 0, 1, 2, 3 corresponding to 0, 90, 180, 270 degrees
            return int(match.group(1)) * 90

        # Fallback to 0
        return 0

    def get_abi(self) -> str:
        """Gets the device ABI (e.g., arm64-v8a, x86_64)."""
        return self.adb.shell("getprop ro.product.cpu.abi").stdout.strip()

    def get_sdk(self) -> str:
        """Gets the device Android SDK version (e.g., 30)."""
        return self.adb.shell("getprop ro.build.version.sdk").stdout.strip()

    def install(self, local_bin_path: str, local_so_path: str) -> bool:
        """
        Installs the minicap binaries and library on the device.
        :param local_bin_path: Path to the minicap binary on the local computer.
        :param local_so_path: Path to the minicap.so library on the local computer.
        """
        info(f"Installing minicap to {self.work_dir}...")
        self.adb.shell(f"mkdir -p {self.work_dir}")

        # Push binary and library
        res1 = self.adb.adb.shell(f"push {local_bin_path} {self.bin_path}")
        res2 = self.adb.adb.shell(f"push {local_so_path} {self.so_path}")

        # Note: AdbDevice.adb is the underlying AdbDevice instance if it exists.
        # Wait, I should use the standard adb push command.
        # Looking at AdbDevice implementation, it doesn't have a direct push.
        # I'll use os.system or subprocess for adb push if needed,
        # or just rely on the user to push them for now if I don't want to mess with adb_pywrapper.

        # Actually, let's just use the subprocess to run adb push directly for simplicity.
        import subprocess

        try:
            subprocess.run(
                ["adb", "-s", self.device_id, "push", local_bin_path, self.bin_path],
                check=True,
            )
            subprocess.run(
                ["adb", "-s", self.device_id, "push", local_so_path, self.so_path],
                check=True,
            )
            self.adb.shell(f"chmod 777 {self.bin_path}")
            info("Minicap installed successfully.")
            return True
        except Exception as e:
            error(f"Failed to install minicap: {e}")
            return False

    def is_installed(self) -> bool:
        """
        Checks if minicap is correctly installed and runnable.
        Verifies existence, permissions, and library linkage.
        Caches the result after the first check.
        """
        if self._is_installed is not None:
            return self._is_installed

        # 1. Basic file existence check
        res = self.adb.shell(f"ls {self.bin_path} {self.so_path}")
        if not res.success or len(res.stdout.strip().splitlines()) < 2:
            self._is_installed = False
            return False

        # 2. Try to run minicap with -i (info) to check if it's runnable and libraries are okay
        # -i flag returns JSON info about the display and exits
        test_cmd = f"LD_LIBRARY_PATH={self.work_dir} {self.bin_path} -i"
        test_res = self.adb.shell(test_cmd)

        # If it returns a JSON starting with '{', it's likely working
        self._is_installed = test_res.success and test_res.stdout.strip().startswith(
            "{"
        )
        return self._is_installed

    def check_installation(self):
        """Prints diagnostic information about the minicap installation."""
        debug(f"===== Minicap Installation Check (Device: {self.device_id}) =====")
        abi = self.get_abi()
        sdk = self.get_sdk()
        debug(f"Device ABI: {abi}")
        debug(f"Device SDK: {sdk}")

        bin_exists = self.adb.shell(f"ls {self.bin_path}").success
        so_exists = self.adb.shell(f"ls {self.so_path}").success

        debug(f"Executable exists ({self.bin_path}): {'YES' if bin_exists else 'NO'}")
        debug(f"Shared library exists ({self.so_path}): {'YES' if so_exists else 'NO'}")

        if bin_exists and so_exists:
            test_cmd = f"LD_LIBRARY_PATH={self.work_dir} {self.bin_path} -i"
            test_res = self.adb.shell(test_cmd)
            if test_res.success and test_res.stdout.strip().startswith("{"):
                debug("Minicap status: OK (Runnable)")
                debug(f"Display Info: {test_res.stdout.strip()}")
            else:
                error("Minicap status: ERROR (Failed to run)")
                error(f"Error output: {test_res.stdout} {test_res.stderr}")
                error(
                    f"Hint: Ensure you pushed the correct minicap.so for SDK {sdk} and ABI {abi}."
                )
        else:
            error("Minicap status: NOT INSTALLED")
            error(
                f"Hint: Push 'minicap' (ABI: {abi}) and 'minicap.so' (SDK: {sdk}, ABI: {abi}) to {self.work_dir}"
            )

    def take_screenshot(self, filename: str) -> bool:
        """
        Takes a single screenshot using minicap and saves it locally.
        :param filename: Local path to save the screenshot (typically .jpg).
        :return: True if successful, False otherwise.
        """
        w, h = self.get_screen_size()
        rotation = self.get_rotation()

        # minicap projection format: <real_w>x<real_h>@<virtual_w>x<virtual_h>/<rotation>
        projection = f"{w}x{h}@{w}x{h}/{rotation}"

        # LD_LIBRARY_PATH is required for the shared library
        # -s flag takes a single snapshot and outputs to stdout
        cmd = f"LD_LIBRARY_PATH={self.work_dir} {self.bin_path} -P {projection} -s"

        # Using adb_pywrapper's shell method which allows local redirection if shell=True
        # adb shell LD_LIBRARY_PATH=... /data/local/tmp/minicap -P ... -s > local_file.jpg
        result = self.adb.shell(f"{cmd} > {filename}")

        return (
            result.success
            and os.path.exists(filename)
            and os.path.getsize(filename) > 0
        )

    def screenshot(self) -> str:
        """
        Captures a screenshot using minicap and returns the local filename.
        Raises RuntimeError if minicap is not available or fails.
        :return: The local path to the saved screenshot.
        """
        os.makedirs("screenshots", exist_ok=True)
        timestamp = int(time())
        pid = os.getpid()
        filename = f"screenshots/{timestamp}_{pid}.jpg"

        if not self.is_installed():
            raise RuntimeError("Minicap is not installed or functional. Exiting.")

        if not self.take_screenshot(filename):
            raise RuntimeError("Minicap failed to capture screenshot. Exiting.")

        return filename
