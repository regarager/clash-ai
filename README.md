# Clash Royale Bot
## Setup

### Dependencies

- An Android emulator using [ADB](https://developer.android.com/tools/adb) with Clash Royale installed
- The [`uv`](https://docs.astral.sh/uv/) package manager

### Installation

1. Setup a virtual environment and activate it: `uv venv && source .venv/bin/activate`
2. Install `pip` dependencies: `uv sync`
3. Download the latest version of `best.pt` from [Releases](https://github.com/regarager/clash-ai/releases) and save it to `vision/best.pt`.
4. Run the program with `python main.py`.

**Note: the emulator should be run with size 1432x1736** to ensure that positions work properly. This can be set with the command
```bash
adb shell wm size 1432x1736
```

## Faster Screenshots with Minicap

By default, the bot uses `adb screencap`, which can be slow. To significantly improve screenshot speed, you can install `minicap`.

### Installation Steps

1.  **Identify your device ABI and SDK**:
    Run the setup script:
    ```bash
    python setup/install_minicap.py
    ```

2.  **Download the correct binaries**:
    Go to the [DeviceFarmer/minicap](https://github.com/DeviceFarmer/minicap) repository and download:
    - `bin/<your-abi>/minicap`
    - `shared/android-<your-sdk>/<your-abi>/minicap.so`

3.  **Push the files to your device**:
    ```bash
    adb push bin/<your-abi>/minicap /data/local/tmp/
    adb push shared/android-<your-sdk>/<your-abi>/minicap.so /data/local/tmp/
    adb shell chmod 777 /data/local/tmp/minicap
    ```

4.  **Verify the installation**:
    Run the setup script again:
    ```bash
    python setup/install_minicap.py
    ```

Once installed, the bot will automatically detect and use `minicap` for faster screen capture.
