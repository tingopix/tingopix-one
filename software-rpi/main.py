# Tingopix — An open DIY film scanner for 8mm, Regular 8, Super 8, and 16mm.
# Sensel-native, archival-first.
#
# Copyright (c) 2026 Pablo Miliani / Tingopix
# Project: https://github.com/tingopix/tingopix-one
# Website: https://tingopix.github.io
#
# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See LICENSE file in the project root for full text.

import os

# Set the Numba on-disk cache location before any numba import (here or in a
# child module) so every process shares one cache. Derived from __file__ so it
# follows the install location; child processes inherit it via the environment.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'numba_cache')
os.environ['NUMBA_CACHE_DIR'] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)

"""
Main orchestration entry point.

Sets up shared memory, launches all processor modules as separate processes
(each pinned to an assigned core), then runs the GTK4 GUI in the main process
as a blocking call. On GUI close, signals a graceful shutdown to the workers
and tears down shared memory.

Shared memory:
    Reads:  -
    Writes: flags[SHUTDOWN_REQUESTED]

Interacts with:
    shm_service (setup/cleanup), mp_setup (start/stop processes),
    mod_gui (blocking GUI main loop)
"""

import time
import psutil

from modules.shm_service import setup_shared_memory, cleanup_shared_memory
from modules.memory.mp_setup import start_all_processes, stop_all_processes
from tpx_logger import log, LogTag

import modules.mod_gui as mod_gui


def main():
    log(LogTag.MAIN, "=== TingoPix ===")

    # Pin the main process (which runs the GUI) to core 0 so the worker
    # processes on other cores are not preempted by GUI event handling.
    gui_core = 0
    try:
        p = psutil.Process()
        p.cpu_affinity([gui_core])
        log(LogTag.MAIN, f"Main process (GUI) bound to core {gui_core}")
    except Exception as e:
        log(LogTag.MAIN, f"Failed to set main process affinity: {e}", level="WARN")

    try:
        # Step 1: Setup shared memory
        setup_shared_memory()

        # Step 2: Start all processor modules (NOT including GUI)
        processes = start_all_processes()

        log(LogTag.MAIN, "All processes started — GUI starting...")

        # Step 3: Run GUI in main process (blocking call)
        try:
            mod_gui.main()
        except KeyboardInterrupt:
            log(LogTag.MAIN, "Shutdown requested via Ctrl+C")

        # When GUI closes, proceed to shutdown
        log(LogTag.MAIN, "GUI closed — initiating shutdown...")

        # Step 4: Set shutdown flag to let processes cleanup gracefully
        from modules.memory.shm_manager import get_array
        from modules.memory.shm_indexing import FlagIndex
        flags = get_array("flags")
        if flags is not None:
            flags[FlagIndex.SHUTDOWN_REQUESTED] = True
            log(LogTag.MAIN, "Shutdown flag set — waiting for processes...")
            time.sleep(2)

        # Step 5: Stop processes
        stop_all_processes(processes)

    except Exception as e:
        log(LogTag.MAIN, f"System error: {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        # Step 6: Cleanup shared memory
        cleanup_shared_memory()

    log(LogTag.MAIN, "System shutdown complete.")


if __name__ == "__main__":
    main()

# ============================================================================
# QUICK REFERENCE: How to add a new shared array
# ============================================================================
# 1. Edit modules/memory/shm_config.py and add to MEMORY_SPECS:
#    "my_array": MemorySpec("shared_my_array", (100, 200), np.float32),
#    (First arg is the OS-level shared-memory name; the dict key is how you
#     look it up below. They need not match.)
#
# 2. Use in any module:
#    from modules.memory.shm_manager import get_array
#    my_array = get_array("my_array")
#    if my_array is not None:
#        my_array[0, 0] = 3.14
# ============================================================================