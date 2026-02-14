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

**Note: the emulator should be run with size 1432x1736** to ensure that positions work properly.
