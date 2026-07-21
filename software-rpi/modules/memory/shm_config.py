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
Shared memory specifications.

Single source of truth for all shared-memory array specifications: shared name,
shape, and dtype for every array in the system. Consumed by shm_manager (to
attach arrays) and shm_service (to create and tear them down).

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -
"""

from typing import NamedTuple
import numpy as np

#Shapes
(blue_height, blue_width) = 1520,384
shape_blue_aoi = (blue_height, blue_width)    # Slice for sprocket
shape_raw_16 = (raw_16_height, raw_16_width) = 3040,4064 
shape_flags = (128,)
shape_sprocket_position = (2,2,4) # Index as Frame, Hole, Detected Value
shape_rgb_8 = (380,508,3) # Used for displaying 1/8 image of capture
shape_wvf_8 = (275,508,4) # Note it is 4 dimensions to make it compatible with GTK Cairo.

# Ring buffer for stacked capture (deep ring: write all frames first, then
# accumulate in a single batched pass for cache efficiency)
RAW_STACK_DEPTH = 16
shape_raw_stack_16 = (RAW_STACK_DEPTH, raw_16_height, raw_16_width)


# Add new shape definitions for string arrays
shape_string_data = (10, 260)  # 10 strings, max 260 chars each
shape_string_lengths = (10,)   # Length of each string

class MemorySpec(NamedTuple):
    name: str      # Shared memory name
    shape: tuple   # Array shape
    dtype: type    # Numpy dtype
    
    @property
    def size_bytes(self) -> int:
        """Calculate total size in bytes"""
        return int(np.prod(self.shape) * self.dtype().itemsize)

# Define all shared memory specifications here
MEMORY_SPECS = {
    "flags": MemorySpec("shared_flags", (shape_flags), np.bool_),

    "shm_display_aoi_0": MemorySpec("display_aoi_0", (blue_width // 2, blue_height // 2,3 ), np.uint8), # shape transposed to display sideways
    "shm_display_aoi_1": MemorySpec("display_aoi_1", (blue_width // 2, blue_height // 2,3 ), np.uint8), # shape transposed to display sideways

    "shm_blue0_8": MemorySpec("blue0_8", (shape_blue_aoi), np.uint8),
    "shm_blue1_8": MemorySpec("blue1_8", (shape_blue_aoi), np.uint8),
    "shm_raw0_16": MemorySpec("raw0_16", (shape_raw_16), np.uint16),
    "shm_raw1_16": MemorySpec("raw1_16", (shape_raw_16), np.uint16),
    "shm_raw_buffer_16": MemorySpec("shm_raw_buffer", (shape_raw_16), np.uint16),
    "shm_raw_capture_16": MemorySpec("shm_raw_capture", (shape_raw_16), np.uint16),
    "shm_raw_stack_16": MemorySpec("shm_raw_stack", shape_raw_stack_16, np.uint16),
    "shm_rgb_capture_16": MemorySpec("shm_rgb_capture", (raw_16_height,raw_16_width,3), np.uint16),
    "shm_rgb0_8": MemorySpec("rgb0_8", (shape_rgb_8), np.uint8),
    "shm_rgb1_8": MemorySpec("rgb1_8", (shape_rgb_8), np.uint8),
    "shm_wvf0_8": MemorySpec("wvf0_8", (shape_wvf_8), np.uint8),
    "shm_wvf1_8": MemorySpec("wvf1_8", (shape_wvf_8), np.uint8),
    "shm_string_data": MemorySpec("string_data", shape_string_data, np.uint8),
    "shm_string_lengths": MemorySpec("string_lengths", shape_string_lengths, np.uint32),
    "shm_sprocket_position_blue0": MemorySpec("sprocket_position_blue0", (shape_sprocket_position), np.uint16),
    "shm_sprocket_position_blue1": MemorySpec("sprocket_position_blue1", (shape_sprocket_position), np.uint16),
    "shm_float_var": MemorySpec("sm_floatvar", (16,), np.float32),
    "shm_uint16_var": MemorySpec("sm_uint16var", (32,), np.uint16),
    "shm_uint32_var": MemorySpec("sm_uint32var", (16,), np.uint32),

    # Add new arrays here
}

def get_spec(key: str) -> MemorySpec:
    """Get memory specification by key"""
    if key not in MEMORY_SPECS:
        raise KeyError(f"No memory specification found for '{key}'")
    return MEMORY_SPECS[key]

def list_all_specs():
    """List all configured memory arrays (no-op: summary printed by shm_service)"""
    pass