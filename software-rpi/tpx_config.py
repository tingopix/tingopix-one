# Tingopix — An open DIY film scanner for 8mm, Regular 8, Super 8, and 16mm.
# Sensel-native, archival-first.
#
# Copyright (c) 2026 Pablo Miliani / Tingopix
# Project: https://github.com/tingopix/tingopix-one
# Website: https://tingopix.github.io
#
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See LICENSE file in the project root for full text.

"""
User configuration.

Single source of truth for all user-adjustable defaults. Edit this file to
change behavior without modifying module code.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -
"""

# ----------------------------------------------------------------------------
# Hardware Mode
# Selects the hardware adapter used by the Serial module.
#   "TINGOPIX" — TingoPix scanner with Pico controller (default)
#   "CUSTOM"   — Third-party scanner, see CustomAdapter in mod_serial.py
# ----------------------------------------------------------------------------
HARDWARE_MODE: str = "TINGOPIX"

# ----------------------------------------------------------------------------
# Serial Port
# Device path for the Pico controller. The default suits a Raspberry Pi with
# no other USB serial devices attached.
#
# If the connection fails, list the available ports with:
#     python3 -m serial.tools.list_ports -v
#
# The Pico appears with vendor ID 0x2E8A (Raspberry Pi), typically described
# as "Pico - Board CDC". Note that /dev/ttyS0 is the Pi's on-board UART and is
# never the scanner. ACM numbering follows USB enumeration order and may change
# between boots when other USB serial devices are present.
# ----------------------------------------------------------------------------
SERIAL_PORT: str = "/dev/ttyACM0"

# ----------------------------------------------------------------------------
# Motion Behavior
# When True, steppers are automatically disabled before the serial port
# disconnects (e.g. on exit). When False, steppers are left as-is on disconnect.
# ----------------------------------------------------------------------------
DISABLE_STEPPERS_ON_DISCONNECT: bool = False

# ----------------------------------------------------------------------------
# Exposure and Light Defaults
# Factory values used on first run, and whenever a saved preset omits a key.
# Once a preset has been saved from the GUI it takes precedence over these.
#
# Exposures are in microseconds. LED values are 12-bit DAC codes (0x000-0xFFF)
# and are specific to your lamp, diffuser, and optical path — expect to
# recalibrate them for your own hardware rather than using these as-is.
# ----------------------------------------------------------------------------
DEFAULT_EXP_DET: int = 4800
DEFAULT_EXP_RGB: int = 4800
DEFAULT_EXP_R:   int = 4800
DEFAULT_EXP_G:   int = 4800
DEFAULT_EXP_B:   int = 4800

DEFAULT_LED_DET: int = 0x380
DEFAULT_LED_RGB: int = 0x638
DEFAULT_LED_R:   int = 0x890
DEFAULT_LED_G:   int = 0x630
DEFAULT_LED_B:   int = 0x99C

# Frames combined per capture. 1 disables stacking.
DEFAULT_STACKING: int = 1

# ----------------------------------------------------------------------------
# Digital Gain Defaults
# R/B digital gains applied during debayering (display and file save).
# Range: 0.0 – 5.0  |  Step: 0.005
# ----------------------------------------------------------------------------
DEFAULT_R_DIG_GAIN: float = 1.665
DEFAULT_B_DIG_GAIN: float = 1.845

# ----------------------------------------------------------------------------
# Colour Correction Matrix
# Selects the CCM from the IMX477 calibration table by illuminant temperature.
# Available entries: 2000-3600 K in 200 K steps, then 4100-8600 K in 500 K
# steps. Values between entries snap to the nearest available temperature.
#
# NOTE: DEFAULT_R_DIG_GAIN / DEFAULT_B_DIG_GAIN were calibrated against this
# matrix. Changing the temperature requires re-deriving those gains against a
# neutral reference.
# ----------------------------------------------------------------------------
CCM_COLOR_TEMP: int = 5100

# ----------------------------------------------------------------------------
# Black Level Correction
# Sensor black level per channel, expressed in the 16-bit scale.
# IMX477: 4096 in 16-bit == 256 at 12-bit.
#
# Nominally uniform across the CFA. Leave the three values equal unless a
# per-channel pedestal has actually been measured on your unit.
# ----------------------------------------------------------------------------
BLACK_LEVEL_R: int = 4096
BLACK_LEVEL_G: int = 4096
BLACK_LEVEL_B: int = 4096

# ----------------------------------------------------------------------------
# File Saving Defaults
# DEFAULT_FILEPATH is the folder shown when the app starts; it can be changed
# at any time from the File Save panel. A leading "~" expands to the current
# user's home directory. The folder is created on first save if missing.
#
# Scanning produces large files, so an external drive is usually preferable:
#     DEFAULT_FILEPATH = "/media/pi/my_drive/scans"
# ----------------------------------------------------------------------------
DEFAULT_FILEPATH: str = "~/mnt/xt12tb/test_frames"
# DEFAULT_FILEPATH: str = "~/tingopix_scans"
DEFAULT_PREFIX:   str = "tpx"