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
Shared memory service

Allocates and tears down every shared-memory array defined in shm_config.
Creates each block from MEMORY_SPECS, zero-fills it, and registers an atexit
handler so buffers are unlinked on shutdown. Exposes a module-level singleton
so all processes share one setup/cleanup coordinator.

Shared memory:
    Reads:  MEMORY_SPECS (allocation specs, not runtime flags)
    Writes: all arrays (creates and zero-fills the entire backing store)

Interacts with:
    shm_config (MEMORY_SPECS), tpx_logger (log/LogTag)
"""
from multiprocessing import shared_memory
import numpy as np
import atexit
from modules.memory.shm_config import MEMORY_SPECS
from tpx_logger import log, LogTag

class SharedService:
    def __init__(self):
        self._shm_objects = {}
        atexit.register(self.cleanup)
    
    def setup_shared_memory(self):
        """Create every shared-memory array from MEMORY_SPECS and zero-fill it."""
        log(LogTag.SHM, "Setting up shared memory...")

        try:
            total_bytes = 0
            for key, spec in MEMORY_SPECS.items():
                shm = shared_memory.SharedMemory(
                    name=spec.name,
                    create=True,
                    size=spec.size_bytes
                )

                array = np.ndarray(spec.shape, dtype=spec.dtype, buffer=shm.buf)
                array.fill(0)

                self._shm_objects[spec.name] = shm
                total_bytes += spec.size_bytes

            total_mb = total_bytes / (1024 * 1024)
            log(LogTag.SHM, f"{len(self._shm_objects)} arrays created — {total_mb:.1f} MB total")

        except FileExistsError:
            log(LogTag.SHM, "Shared memory already exists — cleaning up and retrying...", level="WARN")
            self.cleanup()
            self.setup_shared_memory()
        except Exception as e:
            log(LogTag.SHM, f"Setup error: {e}", level="ERR")
            raise

    def cleanup(self):
        """Close and unlink every allocated shared-memory block."""
        if not self._shm_objects:
            return
        log(LogTag.SHM, "Cleaning up shared memory...")

        for name, shm in self._shm_objects.items():
            try:
                shm.close()
                shm.unlink()
                log(LogTag.SHM, f"  ✓ Cleaned up {name}")
            except Exception as e:
                log(LogTag.SHM, f"  ✗ Error cleaning {name}: {e}", level="ERR")

        self._shm_objects.clear()
        log(LogTag.SHM, "Shared memory cleanup complete.")

_service = None

def get_service():
    global _service
    if _service is None:
        _service = SharedService()
    return _service

def setup_shared_memory():
    get_service().setup_shared_memory()

def cleanup_shared_memory():
    global _service
    if _service:
        _service.cleanup()
        _service = None