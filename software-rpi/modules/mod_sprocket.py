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
Sprocket processor

Prepares the blue-channel AOI slice used for sprocket-hole detection. On each
new raw frame, extracts left/right vertical blue slices from the ping-pong raw
buffers, downsamples them, and writes both a grayscale AOI display and the
blue-slice buffer, then raises a detection request for the detector module.

Shared memory:
    Reads:  MOD_READY_DEBAYER, MOD_READY_ALL, SHUTDOWN_REQUESTED,
            SPROCKET_HALT, RQST_DBY_B0, RQST_DBY_B1, AOI_POSITION,
            raw0_16, raw1_16
    Writes: MOD_READY_SPROCKET, RQST_DBY_B0/B1 (clears), RQST_SPK_B0,
            RQST_SPK_B1, AOI_POSITION (init), blue0_8, blue1_8,
            display_aoi_0, display_aoi_1

Interacts with:
    mod_capture (RQST_DBY_B0/B1), mod_detector (RQST_SPK_B0/B1),
    mod_gui (SPROCKET_HALT, AOI_POSITION), shm_manager (arrays)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, Uint16Index
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

import time
import numpy as np
from numba import njit


@staticmethod
@njit(cache=True)
def debayer_blue_slice_16_fast(raw_16, blue_aoi, display,position_x = 0):
    height, width = raw_16.shape
    raw_blue_width = 384
    adjusted_width = raw_blue_width - 128
    
    # Left slice - vectorized
    blue_aoi[:, :adjusted_width] = (raw_16[0:height:2, position_x:(adjusted_width * 2)+position_x:2] >> 4).astype(np.uint8)
    
    # Right slice - vectorized
    blue_aoi[:, raw_blue_width-128:raw_blue_width] = (raw_16[0:height:2, width-257:width-1:2] >> 4).astype(np.uint8)
    
    # Transpose
    blue_t = blue_aoi.T
    
    # Downsample by 2x2 averaging (add neighboring pixels and divide by 4)
    h, w = blue_t.shape
    blue_t_small = ((blue_t[0:h:2, 0:w:2].astype(np.uint16) + 
                     blue_t[1:h:2, 0:w:2].astype(np.uint16) + 
                     blue_t[0:h:2, 1:w:2].astype(np.uint16) + 
                     blue_t[1:h:2, 1:w:2].astype(np.uint16)) >> 2).astype(np.uint8)
    
    # Get dimensions of downsampled image
    h_small, w_small = blue_t_small.shape
    
    # Assign to corresponding portion of display
    display[:h_small, :w_small, 0] = blue_t_small
    display[:h_small, :w_small, 1] = blue_t_small
    display[:h_small, :w_small, 2] = blue_t_small

class SprocketProcessor:
    def __init__(self):
        log(LogTag.SPKT, "Initializing...")
        
        # Get arrays using config keys
        self.flags = get_array("flags")
        self.raw0_16 = get_array("shm_raw0_16")
        self.raw1_16 = get_array("shm_raw1_16")
        self.blue0_8 = get_array("shm_blue0_8")
        self.blue1_8 = get_array("shm_blue1_8")
        self.display_aoi_0 = get_array("shm_display_aoi_0")
        self.display_aoi_1 = get_array("shm_display_aoi_1")
        self.sm_uint16var = get_array("shm_uint16_var")
        
        if self.flags is None or self.raw0_16 is None or self.raw1_16 is None or self.blue0_8 is None or self.blue1_8 is None or self.display_aoi_0 is None or self.display_aoi_1 is None or self.sm_uint16var is None:
            raise RuntimeError("Module Sprocket: Failed to get shared arrays")
        
        self.sm_uint16var[Uint16Index.AOI_POSITION] = 0 # Initialize AOI_POSITION

        self.AOI_position = self.sm_uint16var[Uint16Index.AOI_POSITION]
        log(LogTag.SPKT, "Connected to shared memory")

        #Wait until last module finished compiling
        while not self.flags[FlagIndex.MOD_READY_DEBAYER]:
            time.sleep(0.5)

        log_compile_start(LogTag.SPKT, "blue debayer function")
        compile_start = time.perf_counter()

        debayer_blue_slice_16_fast(self.raw0_16, self.blue0_8, self.display_aoi_0, self.AOI_position)
        debayer_blue_slice_16_fast(self.raw1_16, self.blue1_8, self.display_aoi_1, self.AOI_position)

        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.SPKT, compile_time)
        log(LogTag.SPKT, "Ready")

        #Raised finished compiling
        self.flags[FlagIndex.MOD_READY_SPROCKET] = True

        #Wait until last module finished compiling
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)

    def process(self):
        """Main processing loop"""
        log(LogTag.SPKT, "Starting capture loop...")

        try:
            # debayering Blue AOI
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:  # Check shutdown flag
                if not self.flags[FlagIndex.SPROCKET_HALT]:
                    
                    self.AOI_position = self.sm_uint16var[Uint16Index.AOI_POSITION]

                    if (self.flags[FlagIndex.RQST_DBY_B0]): #Request from Capture to Debayer
                        self.flags[FlagIndex.RQST_DBY_B0] = False 

                        self.flags[FlagIndex.RQST_SPK_B0] = False #Clear Sprocket Detection Requests
                        self.flags[FlagIndex.RQST_SPK_B1] = False #Clear Sprocket Detection Requests


                        debayer_blue_slice_16_fast (self.raw0_16, self.blue0_8, self.display_aoi_0,self.AOI_position)

                        self.flags[FlagIndex.RQST_SPK_B0] = True #Sprocket Detection Request to Detection Module



                    if (self.flags[FlagIndex.RQST_DBY_B1]): #Raw1 Captured
                        self.flags[FlagIndex.RQST_DBY_B1] = False #Cleared by Blue Debayering

                        self.flags[FlagIndex.RQST_SPK_B0] = False #Clear Sprocket Detection Requests
                        self.flags[FlagIndex.RQST_SPK_B1] = False #Clear Sprocket Detection Requests

                        debayer_blue_slice_16_fast (self.raw1_16, self.blue1_8, self.display_aoi_1,self.AOI_position)
    
                        self.flags[FlagIndex.RQST_SPK_B1] = True #Sprocket Detection Request to Detection Module

                time.sleep(0.001)  # Small sleep to avoid busy-wait

        except KeyboardInterrupt:
            log(LogTag.SPKT, "Interrupted by user")
        except Exception as e:
            log(LogTag.SPKT, f"Error — {e}", level="ERR")
        finally:
            log(LogTag.SPKT, "Stopping...")

def main():
    """Main function for capture module"""
    processor = None
    try:
        processor = SprocketProcessor()
        processor.process()
    except Exception as e:
        log(LogTag.SPKT, f"Fatal error — {e}", level="ERR")
    finally:
        log(LogTag.SPKT, "Shutting down...")
        cleanup_manager()
        log(LogTag.SPKT, "Cleanup complete")