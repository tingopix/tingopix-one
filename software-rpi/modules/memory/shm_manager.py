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
Shared memory manager.

Per-process manager that connects to the shared-memory regions defined in
shm_config and returns numpy array views by config key. Also provides typed
helpers for reading and writing the shared string array.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -
"""

from multiprocessing import shared_memory
import numpy as np
from typing import Optional, Dict, List
from .shm_config import MEMORY_SPECS, MemorySpec
from tpx_logger import log, LogTag

class MemoryManager:
    def __init__(self):
        self._connections: Dict[str, shared_memory.SharedMemory] = {}
        self._arrays: Dict[str, np.ndarray] = {}
    
    def _connect_to_shared_memory(self, spec: MemorySpec) -> Optional[shared_memory.SharedMemory]:
        """Connect to a shared memory region using spec"""
        if spec.name not in self._connections:
            try:
                self._connections[spec.name] = shared_memory.SharedMemory(name=spec.name)
            except FileNotFoundError:
                log(LogTag.SHM, f"'{spec.name}' not found — ensure setup was called first.", level="ERR")
                return None
        return self._connections[spec.name]
    
    def get_array(self, key: str) -> Optional[np.ndarray]:
        """Get an array by its specification key"""
        if key not in MEMORY_SPECS:
            log(LogTag.SHM, f"No specification found for '{key}'", level="ERR")
            return None
        
        if key not in self._arrays:
            spec = MEMORY_SPECS[key]
            shm = self._connect_to_shared_memory(spec)
            if shm:
                self._arrays[key] = np.ndarray(spec.shape, dtype=spec.dtype, buffer=shm.buf)
        
        return self._arrays.get(key)
    
    def cleanup_connections(self):
        """Clean up this manager's connections"""
        for shm in self._connections.values():
            try:
                shm.close()
            except:
                pass
        self._connections.clear()
        self._arrays.clear()

    def write_string(self, index: int, value: str) -> None:
        """Write a string to the shared string array at the given index."""
        string_data = self.get_array("shm_string_data")
        string_lengths = self.get_array("shm_string_lengths")
        
        if string_data is None or string_lengths is None:
            raise RuntimeError("String arrays not initialized")
        
        num_strings, max_length = string_data.shape
        
        if index < 0 or index >= num_strings:
            raise IndexError(f"Index {index} out of range (0-{num_strings-1})")
        
        # Encode string to bytes
        encoded = value.encode('utf-8')
        
        if len(encoded) > max_length:
            raise ValueError(f"String too long: {len(encoded)} > {max_length}")
        
        # Clear the row first (optional, for cleanliness)
        string_data[index, :] = 0
        
        # Write the string bytes
        string_data[index, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
        
        # Write the actual length
        string_lengths[index] = len(encoded)
    
    def read_string(self, index: int) -> str:
        """Read a string from the shared string array at the given index."""
        string_data = self.get_array("shm_string_data")
        string_lengths = self.get_array("shm_string_lengths")
        
        if string_data is None or string_lengths is None:
            raise RuntimeError("String arrays not initialized")
        
        num_strings = string_data.shape[0]
        
        if index < 0 or index >= num_strings:
            raise IndexError(f"Index {index} out of range (0-{num_strings-1})")
        
        # Get the actual length
        actual_length = int(string_lengths[index])
        
        if actual_length == 0:
            return ""
        
        # Read only the actual bytes and decode
        string_bytes = bytes(string_data[index, :actual_length])
        return string_bytes.decode('utf-8')
    
    def write_all_strings(self, strings: List[str]) -> None:
        """Write multiple strings at once."""
        string_data = self.get_array("shm_string_data")
        if string_data is None:
            raise RuntimeError("String arrays not initialized")
        
        num_strings = string_data.shape[0]
        for i, s in enumerate(strings):
            if i >= num_strings:
                break
            self.write_string(i, s)
    
    def read_all_strings(self) -> List[str]:
        """Read all strings from the shared array."""
        string_data = self.get_array("shm_string_data")
        if string_data is None:
            raise RuntimeError("String arrays not initialized")
        
        num_strings = string_data.shape[0]
        return [self.read_string(i) for i in range(num_strings)]
    
    def clear_string(self, index: int = None) -> None:
        """Clear a specific string or all strings."""
        string_lengths = self.get_array("shm_string_lengths")
        if string_lengths is None:
            raise RuntimeError("String arrays not initialized")
        
        if index is not None:
            string_lengths[index] = 0
        else:
            string_lengths[:] = 0


# Global manager instance per process
_manager = None

def get_manager() -> MemoryManager:
    """Get the global memory manager for this process"""
    global _manager
    if _manager is None:
        _manager = MemoryManager()
    return _manager

# Generic function - use this for any array
def get_array(key: str) -> Optional[np.ndarray]:
    """Get any shared array by its config key"""
    return get_manager().get_array(key)

def cleanup_manager():
    """Clean up the manager connections"""
    global _manager
    if _manager:
        _manager.cleanup_connections()
        _manager = None

def write_string(index: int, value: str) -> None:
    """Write a string to shared memory."""
    return get_manager().write_string(index, value)

def read_string(index: int) -> str:
    """Read a string from shared memory."""
    return get_manager().read_string(index)

def write_all_strings(strings: List[str]) -> None:
    """Write multiple strings to shared memory."""
    return get_manager().write_all_strings(strings)

def read_all_strings() -> List[str]:
    """Read all strings from shared memory."""
    return get_manager().read_all_strings()

def clear_string(index: int = None) -> None:
    """Clear string(s) in shared memory."""
    return get_manager().clear_string(index)