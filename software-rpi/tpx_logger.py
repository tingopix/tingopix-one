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
Console logger.

Centralised, timestamped logging utility imported directly by all TingoPix
modules. Provides tagged log lines and Numba compile-start/done helpers with
per-module ETA hints.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -

Usage:
    from tpx_logger import log, LogTag

    log(LogTag.CAP, "Starting capture loop...")
    log(LogTag.CAP, "Error - {e}", level="ERR")

Output format:
    [HH:MM:SS] [TAG   ] Message
    [HH:MM:SS] [ERR   ] [TAG   ] Message
"""

from datetime import datetime

# ----------------------------------------------------------------------------
# Module tags  — fixed 5-char width, printed as [TAG  ]
# ----------------------------------------------------------------------------
class LogTag:
    MAIN  = "MAIN"
    SHM   = "SHM"
    SETUP = "SETUP"
    DFLTS = "DFLTS"
    CAP   = "CAP"
    SPKT  = "SPKT"
    DET   = "DET"
    DEBAY = "DEBAY"
    WFM   = "WFM"
    RCAP  = "RCAP"
    FSAVE = "FSAVE"
    SER   = "SER"
    CTRL  = "CTRL"
    GUI   = "GUI"
    AOI   = "AOI"

# ----------------------------------------------------------------------------
# Warm-up ETA hints (first-run / cached times from profiling)
# Shown before each Numba compilation starts.
# ----------------------------------------------------------------------------
COMPILE_ETA = {
    LogTag.DEBAY: ("~32s", "~1s"),
    LogTag.SPKT:  ("~15s", "~1s"),
    LogTag.RCAP:  ("~18s", "~1s"),
    LogTag.CAP:   ("~11s", "~1s"),
    LogTag.DET:   ("~4s",  "~1s"),
    LogTag.WFM:   ("~6s",  "~1s"),
    LogTag.FSAVE: ("~3s",  "~1s"),
    LogTag.AOI:   ("~17s", "~1s"),
}

# ----------------------------------------------------------------------------
# Core log function
# ----------------------------------------------------------------------------
TAG_WIDTH = 5

def log(tag: str, message: str, level: str = "INFO") -> None:
    """
    Print a timestamped, tagged log line.

    Args:
        tag:     One of the LogTag constants.
        message: The message string (already formatted by caller).
        level:   "INFO" (default), "WARN", or "ERR".
    """
    ts  = datetime.now().strftime("%H:%M:%S")
    tag_str = f"[{tag:<{TAG_WIDTH}}]"

    if level == "WARN":
        print(f"[{ts}] [WARN ] {tag_str} {message}", flush=True)
    elif level == "ERR":
        print(f"[{ts}] [ERR  ] {tag_str} {message}", flush=True)
    else:
        print(f"[{ts}] {tag_str} {message}", flush=True)


def log_compile_start(tag: str, description: str = "JIT function") -> None:
    """
    Print a compilation-start line with first-run / cached ETA hint.

    Args:
        tag:         Module tag (used to look up ETA).
        description: Short label for what is being compiled.
    """
    if tag in COMPILE_ETA:
        first_run, cached = COMPILE_ETA[tag]
        eta_hint = f"  (first run: {first_run} | cached: {cached})"
    else:
        eta_hint = ""
    log(tag, f"Compiling {description}...{eta_hint}")


def log_compile_done(tag: str, elapsed_ms: float) -> None:
    """
    Print a compilation-complete line with elapsed time.

    Args:
        tag:        Module tag.
        elapsed_ms: Elapsed time in milliseconds.
    """
    log(tag, f"Compilation complete ({elapsed_ms:.0f}ms) ✓")