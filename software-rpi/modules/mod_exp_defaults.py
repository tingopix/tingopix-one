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
Exposure and Light Defaults — factory values and preset save/load.

Single source of truth for the scanner's default exposure, LED, stacking, and
digital-gain values. Provides idempotent in-memory initialization, JSON preset
save/load (used by the GUI's save/load controls), and startup auto-load of
exposure_defaults.json. All values are written directly into the shared arrays
passed in by the caller; this module holds no state of its own.

Shared memory:
    Reads:  Uint32: EXPOSURE (init-guard only); on save — EXP_DET, EXP_RGB,
            EXP_R, EXP_G, EXP_B; Uint16: LED_DET, LED_RGB, LED_R, LED_G,
            LED_B; Float: R_DIG_GAIN, B_DIG_GAIN
    Writes: Uint32: EXPOSURE, RAW0_EXPOSURE, RAW1_EXPOSURE, EXP_DET, EXP_RGB,
            EXP_R, EXP_G, EXP_B; Uint16: LED_OFF, LED_DET, LED_RGB, LED_R,
            LED_G, LED_B, STACKING_SETTING; Float: R_DIG_GAIN, B_DIG_GAIN

Interacts with:
    Called directly (not via flags) by GUI/startup code that passes in the
    shared arrays; no cross-process flag handshakes of its own.
"""
import json
from datetime import datetime
from pathlib import Path

from modules.memory.shm_indexing import Uint16Index, Uint32Index, FloatIndex
from tpx_config import (DEFAULT_R_DIG_GAIN, DEFAULT_B_DIG_GAIN,
                        DEFAULT_EXP_DET, DEFAULT_EXP_RGB,
                        DEFAULT_EXP_R, DEFAULT_EXP_G, DEFAULT_EXP_B,
                        DEFAULT_LED_DET, DEFAULT_LED_RGB,
                        DEFAULT_LED_R, DEFAULT_LED_G, DEFAULT_LED_B,
                        DEFAULT_STACKING)
from tpx_logger import log, LogTag


def initialize_defaults(uint32_var, uint16_var, float_var=None, force=False):
    """Write the factory exposure, LED, stacking, and gain defaults into the
    shared arrays. Idempotent and safe to call repeatedly: unless force is set,
    it only writes when EXPOSURE is still 0 (uninitialized), returning True when
    it writes and False when it finds values already in place."""
    if not force and uint32_var[Uint32Index.EXPOSURE] != 0:
        log(LogTag.DFLTS, "Already initialized, skipping")
        return False

    # Factory values come from tpx_config.py — edit them there, not here.
    # Every module reads these from shared memory, so whatever is written
    # below becomes the values the whole system uses.

    # Exposure defaults (microseconds)
    # EXPOSURE and the two RAW*_EXPOSURE trackers hold the camera's current
    # exposure rather than a user setting; they are primed to the detection
    # at-rest exposure so the first capture sees no phantom transition.
    uint32_var[Uint32Index.EXPOSURE] = DEFAULT_EXP_DET      # Master/current exposure
    uint32_var[Uint32Index.RAW0_EXPOSURE] = DEFAULT_EXP_DET # Buffer 0 actual exposure
    uint32_var[Uint32Index.RAW1_EXPOSURE] = DEFAULT_EXP_DET # Buffer 1 actual exposure
    uint32_var[Uint32Index.EXP_DET] = DEFAULT_EXP_DET       # Detection exposure
    uint32_var[Uint32Index.EXP_RGB] = DEFAULT_EXP_RGB       # RGB combined exposure
    uint32_var[Uint32Index.EXP_R] = DEFAULT_EXP_R           # Red channel exposure
    uint32_var[Uint32Index.EXP_G] = DEFAULT_EXP_G           # Green channel exposure
    uint32_var[Uint32Index.EXP_B] = DEFAULT_EXP_B           # Blue channel exposure

    # Light defaults (12-bit DAC values: 0x000 to 0xFFF)
    uint16_var[Uint16Index.LED_OFF] = 0x000                 # Lights off (dark)
    uint16_var[Uint16Index.LED_DET] = DEFAULT_LED_DET       # Detection light
    uint16_var[Uint16Index.LED_RGB] = DEFAULT_LED_RGB       # RGB combined light
    uint16_var[Uint16Index.LED_R] = DEFAULT_LED_R           # Red channel light
    uint16_var[Uint16Index.LED_G] = DEFAULT_LED_G           # Green channel light
    uint16_var[Uint16Index.LED_B] = DEFAULT_LED_B           # Blue channel light

    # Stacking default
    uint16_var[Uint16Index.STACKING_SETTING] = DEFAULT_STACKING

    # Digital gain defaults
    if float_var is not None:
        float_var[FloatIndex.R_DIG_GAIN] = DEFAULT_R_DIG_GAIN
        float_var[FloatIndex.B_DIG_GAIN] = DEFAULT_B_DIG_GAIN

    log(LogTag.DFLTS, f"Defaults set — EXP={uint32_var[Uint32Index.EXPOSURE]}µs  LED_DET=0x{uint16_var[Uint16Index.LED_DET]:03X}")

    return True


# Preset Save/Load Functions

# Path resolution: this file is in modules/, go up to project root
MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent
PRESETS_DIR = PROJECT_ROOT / "presets"

def ensure_presets_dir():
    """Create the presets directory if it does not yet exist and return its
    path."""
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    return PRESETS_DIR

def save_preset(uint32_var, uint16_var, filepath, float_var=None, mode="ALL"):
    """Write the current exposure, LED, and (if provided) gain values from
    shared memory to a JSON preset file at filepath, tagged with the given
    mode. Returns True on success, False if writing fails."""
    try:
        ensure_presets_dir()

        preset = {
            "version": "1.2",
            "mode": mode,
            "created": datetime.now().isoformat(),
            "exposure": {
                "EXP_DET": int(uint32_var[Uint32Index.EXP_DET]),
                "EXP_RGB": int(uint32_var[Uint32Index.EXP_RGB]),
                "EXP_R": int(uint32_var[Uint32Index.EXP_R]),
                "EXP_G": int(uint32_var[Uint32Index.EXP_G]),
                "EXP_B": int(uint32_var[Uint32Index.EXP_B])
            },
            "leds": {
                "LED_DET": int(uint16_var[Uint16Index.LED_DET]),
                "LED_RGB": int(uint16_var[Uint16Index.LED_RGB]),
                "LED_R": int(uint16_var[Uint16Index.LED_R]),
                "LED_G": int(uint16_var[Uint16Index.LED_G]),
                "LED_B": int(uint16_var[Uint16Index.LED_B])
            }
        }

        if float_var is not None:
            preset["gains"] = {
                "R_DIG_GAIN": float(float_var[FloatIndex.R_DIG_GAIN]),
                "B_DIG_GAIN": float(float_var[FloatIndex.B_DIG_GAIN]),
            }

        with open(filepath, 'w') as f:
            json.dump(preset, f, indent=2)

        log(LogTag.DFLTS, f"Preset saved to {filepath}")
        return True

    except Exception as e:
        log(LogTag.DFLTS, f"Error saving preset — {e}", level="ERR")
        return False

def load_preset(uint32_var, uint16_var, filepath, float_var=None):
    """Load exposure, LED, and gain values from a JSON preset file into shared
    memory. Missing keys fall back to factory defaults so older preset files
    load cleanly. Returns the preset's mode string on success, False if the
    file is missing, or None if reading fails."""
    try:
        if not Path(filepath).exists():
            log(LogTag.DFLTS, f"Preset file not found — {filepath}", level="WARN")
            return False

        with open(filepath, 'r') as f:
            preset = json.load(f)

        if "exposure" in preset:
            exp = preset["exposure"]
            uint32_var[Uint32Index.EXP_DET] = exp.get("EXP_DET", DEFAULT_EXP_DET)
            uint32_var[Uint32Index.EXP_RGB] = exp.get("EXP_RGB", DEFAULT_EXP_RGB)
            uint32_var[Uint32Index.EXP_R] = exp.get("EXP_R", DEFAULT_EXP_R)
            uint32_var[Uint32Index.EXP_G] = exp.get("EXP_G", DEFAULT_EXP_G)
            uint32_var[Uint32Index.EXP_B] = exp.get("EXP_B", DEFAULT_EXP_B)
            # Prime EXPOSURE (the loop's current-exposure tracker) to the
            # detection at-rest exposure so the first capture does not see a
            # phantom 0->EXP_DET transition (~15 frames / ~1.9s). Matches the
            # end-of-sequence restore in mod_raw_capture.
            uint32_var[Uint32Index.EXPOSURE] = uint32_var[Uint32Index.EXP_DET]

        if "leds" in preset:
            leds = preset["leds"]
            uint16_var[Uint16Index.LED_DET] = leds.get("LED_DET", DEFAULT_LED_DET)
            uint16_var[Uint16Index.LED_RGB] = leds.get("LED_RGB", DEFAULT_LED_RGB)
            uint16_var[Uint16Index.LED_R] = leds.get("LED_R", DEFAULT_LED_R)
            uint16_var[Uint16Index.LED_G] = leds.get("LED_G", DEFAULT_LED_G)
            uint16_var[Uint16Index.LED_B] = leds.get("LED_B", DEFAULT_LED_B)

        # Falls back to camera_config defaults for old preset files missing "gains"
        if float_var is not None:
            gains = preset.get("gains", {})
            float_var[FloatIndex.R_DIG_GAIN] = gains.get("R_DIG_GAIN", DEFAULT_R_DIG_GAIN)
            float_var[FloatIndex.B_DIG_GAIN] = gains.get("B_DIG_GAIN", DEFAULT_B_DIG_GAIN)

        loaded_mode = preset.get("mode", "ALL")
        log(LogTag.DFLTS, f"Preset loaded from {filepath} (mode={loaded_mode})")
        return loaded_mode

    except Exception as e:
        log(LogTag.DFLTS, f"Error loading preset — {e}", level="ERR")
        return None

def auto_load_defaults(uint32_var, uint16_var, float_var=None):
    """Called at startup to load presets/exposure_defaults.json if it exists,
    returning the loaded mode. If no default preset is found, seed the factory
    defaults instead and return False."""
    ensure_presets_dir()
    default_preset = PRESETS_DIR / "exposure_defaults.json"

    if default_preset.exists():
        log(LogTag.DFLTS, f"Auto-loading {default_preset}")
        return load_preset(uint32_var, uint16_var, default_preset, float_var)
    else:
        log(LogTag.DFLTS, "No exposure_defaults.json found — using factory defaults", level="WARN")
        initialize_defaults(uint32_var, uint16_var, float_var, force=True)
        return False