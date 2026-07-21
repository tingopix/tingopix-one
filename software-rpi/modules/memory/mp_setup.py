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
Process setup and coordination.

Launches all nine processor modules as separate processes, each pinned to its
assigned CPU core, and stops them gracefully on shutdown. Called by main.py.

Shared memory:
    Reads:  -
    Writes: flags[SHUTDOWN_REQUESTED]

Interacts with:
    Starts/stops the processor modules: mod_capture, mod_sprocket,
    mod_detector, mod_debayer, mod_waveform, mod_raw_capture,
    mod_file_saver, mod_serial, mod_controller. Invoked by main.py.
"""

import multiprocessing as mp
import time
from modules.memory.shm_manager import get_array
from modules.memory.shm_indexing import FlagIndex
from tpx_logger import log, LogTag


def _start_with_affinity(process, cores, label):
    """Start a process and set CPU affinity, logging the result."""
    import psutil
    process.start()
    psutil.Process(process.pid).cpu_affinity(cores)
    core_str = ",".join(str(c) for c in cores)
    log(LogTag.SETUP, f"{label} started — core(s) [{core_str}]")


def start_all_processes():
    """Start all processor processes"""
    log(LogTag.SETUP, "Starting processor processes...")

    import modules.mod_capture as mod_capture
    import modules.mod_sprocket as mod_sprocket
    import modules.mod_detector as mod_detector
    import modules.mod_debayer as mod_debayer
    import modules.mod_waveform as mod_waveform
    import modules.mod_raw_capture as mod_raw_capture
    import modules.mod_file_saver as mod_file_saver
    import modules.mod_serial as mod_serial
    import modules.mod_controller as mod_controller

    # Create processes
    p1 = mp.Process(target=mod_capture.main,     name="Capture")
    p2 = mp.Process(target=mod_sprocket.main,    name="Sprocket")
    p3 = mp.Process(target=mod_detector.main,    name="Detector")
    p4 = mp.Process(target=mod_debayer.main,     name="Debayer")
    p5 = mp.Process(target=mod_waveform.main,    name="Waveform")
    p6 = mp.Process(target=mod_raw_capture.main, name="RawCapture")
    p7 = mp.Process(target=mod_file_saver.main,  name="FileSaver")
    p8 = mp.Process(target=mod_serial.main,      name="Serial")
    p9 = mp.Process(target=mod_controller.main,  name="Controller")

    processes = {
        "Capture":    p1,
        "Sprocket":   p2,
        "Detector":   p3,
        "Debayer":    p4,
        "Waveform":   p5,
        "RawCapture": p6,
        "FileSaver":  p7,
        "Serial":     p8,
        "Controller": p9,
    }

    try:
        import psutil
        _start_with_affinity(p1, [3],    "Capture")
        _start_with_affinity(p2, [1],    "Sprocket")
        _start_with_affinity(p3, [2],    "Detector")
        _start_with_affinity(p4, [2],    "Debayer")
        _start_with_affinity(p5, [0],    "Waveform")
        _start_with_affinity(p6, [1],    "RawCapture")
        _start_with_affinity(p7, [1],    "FileSaver")
        _start_with_affinity(p8, [0],    "Serial")
        _start_with_affinity(p9, [0],    "Controller")

    except ImportError:
        log(LogTag.SETUP, "psutil not available — no CPU affinity set", level="WARN")
        for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9]:
            p.start()

    except Exception as e:
        log(LogTag.SETUP, f"Failed to set CPU affinity: {e}", level="WARN")
        for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9]:
            if not p.is_alive():
                p.start()

    return processes


def stop_all_processes(processes):
    """Safely stop all processes with timeout and error handling."""
    log(LogTag.SETUP, "Shutting down processes...")

    # Set shutdown flag
    flags = get_array("flags")
    if flags is not None:
        flags[FlagIndex.SHUTDOWN_REQUESTED] = True

    # Wait for processes to finish gracefully
    for name, p in processes.items():
        if p is None or p._popen is None:
            continue

        if p.is_alive():
            try:
                p.join(timeout=5)
                if p.is_alive():
                    log(LogTag.SETUP, f"{name}: timeout — terminating", level="WARN")
                    p.terminate()
                    p.join(timeout=2)
            except Exception as e:
                log(LogTag.SETUP, f"{name}: error during join — {e}", level="ERR")
                try:
                    p.terminate()
                except:
                    pass
        else:
            log(LogTag.SETUP, f"  {name}: stopped")

    # Force terminate any remaining processes
    for name, p in processes.items():
        if p is not None and p._popen is not None:
            try:
                if p.is_alive():
                    log(LogTag.SETUP, f"{name}: force terminating", level="WARN")
                    p.kill()
                    p.join(timeout=1)
            except Exception as e:
                log(LogTag.SETUP, f"{name}: error during force terminate — {e}", level="ERR")

    log(LogTag.SETUP, "All processes stopped")