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
Display waveform

Renders the RGB monitor preview into a waveform scope for exposure and color
judgement. Offers two layouts — stacked (channels overlaid) and parade (R/G/B
side by side) — writing an 8-bit 4-channel buffer (Cairo-compatible; channel 3
left as alpha/padding) with a peak indicator row. Consumes each debayered
preview frame on request and signals when the matching waveform buffer is
ready.

Shared memory:
    Reads:  MOD_READY_DETECTOR, MOD_READY_ALL, SHUTDOWN_REQUESTED,
            RQST_WVF_0, RQST_WVF_1, WVF_PARADE, rgb0_8, rgb1_8
    Writes: MOD_READY_WAVEFORM, RQST_WVF_0/RQST_WVF_1 (clears),
            READY_WVF0, READY_WVF1, wvf0_8, wvf1_8

Interacts with:
    mod_debayer (RQST_WVF_0/1, rgb0_8/rgb1_8), mod_gui (READY_WVF0/1,
    WVF_PARADE), shm_manager (arrays)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

import time
from numba import njit
import numpy as np


@njit(parallel=True, cache=True)
def process_waveform_stacked(rgb_source, wvf_buffer):
    # Clear the buffer first (including alpha channel)
    wvf_buffer[:, :, :] = 0
    
    for gc in range(0, 3, 1):
        for gy in range(0, 379, 1):
            for gx in range(0, 503, 1):
                # Only write to BGR channels (0, 1, 2), leave channel 3 (alpha/padding) as 0
                if (wvf_buffer[268 - rgb_source[gy][gx][gc]][gx][2-gc] < 160):
                    wvf_buffer[268 - rgb_source[gy][gx][gc]][gx][2-gc] += 16
  

    
    # Peak indicator - write to RGB channels only
    for gx in range(0, 503, 1):
        if ((wvf_buffer[13][gx][0] != 0) or 
            (wvf_buffer[13][gx][1] != 0) or 
            (wvf_buffer[13][gx][2] != 0)):
            wvf_buffer[13][gx][0:3] = 255  # RGB only, leave [3] as 0
    
    return

@njit(parallel=True, cache=True)
def process_waveform_parade(rgb_source, wvf_buffer):
    # Clear the buffer first (including alpha channel)
    wvf_buffer[:, :, :] = 0
    
    for gc in range(0, 3, 1):
        for gy in range(0, 379, 1):
            for gx in range(0, 503, 1):
                # Only write to BGR channels (0, 1, 2), leave channel 3 (alpha/padding) as 0
                if (wvf_buffer[268 - rgb_source[gy][gx][gc]][int(gx/3)+(gc*168)][2-gc] < 160):
                    wvf_buffer[268 - rgb_source[gy][gx][gc]][int(gx/3)+(gc*168)][2-gc] += 8
    
    # Peak indicator - write to RGB channels only
    for gx in range(0, 503, 1):
        if ((wvf_buffer[13][gx][0] != 0) or 
            (wvf_buffer[13][gx][1] != 0) or 
            (wvf_buffer[13][gx][2] != 0)):
            wvf_buffer[13][gx][0:3] = 255  # RGB only, leave [3] as 0
    
    return


class DisplayWaveform:
    def __init__(self):
        log(LogTag.WFM, "Initializing...")
      
        # Get arrays using config keys
        self.flags = get_array("flags")
        self.rgb0_8 = get_array("shm_rgb0_8")
        self.rgb1_8 = get_array("shm_rgb1_8")
        self.wvf0_8 = get_array("shm_wvf0_8")
        self.wvf1_8 = get_array("shm_wvf1_8")

        if (self.flags is None or
            self.rgb0_8 is None or
            self.rgb1_8 is None or
            self.wvf0_8 is None or
            self.wvf1_8 is None):
            raise RuntimeError("Waveform: Failed to get shared arrays")

        log(LogTag.WFM, "Connected to shared memory")

        self.flags[FlagIndex.RQST_WVF_0] = False
        self.flags[FlagIndex.RQST_WVF_1] = False

        #Wait until Previous module finished compiling
        while not self.flags[FlagIndex.MOD_READY_DETECTOR]:
            time.sleep(0.5)

        log_compile_start(LogTag.WFM, "waveform function")
        compile_start = time.perf_counter()

        process_waveform_parade(self.rgb0_8, self.wvf0_8)
        process_waveform_stacked(self.rgb0_8, self.wvf0_8)

        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.WFM, compile_time)

        #Raised finished compiling
        self.flags[FlagIndex.MOD_READY_WAVEFORM] = True

        #Wait until last module finished compiling
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)
        log(LogTag.WFM, "Ready")


    def process(self):
        """Main processing loop"""
        log(LogTag.WFM, "Starting waveform loop...")
          
        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                if self.flags[FlagIndex.RQST_WVF_0]:
                    self.flags[FlagIndex.RQST_WVF_0] = False

                    if self.flags[FlagIndex.WVF_PARADE]:
                        process_waveform_parade(self.rgb0_8,self.wvf0_8)
                    else:
                        process_waveform_stacked(self.rgb0_8,self.wvf0_8)

                    self.flags[FlagIndex.READY_WVF1]=False
                    self.flags[FlagIndex.READY_WVF0]=True

                if self.flags[FlagIndex.RQST_WVF_1]:
                    self.flags[FlagIndex.RQST_WVF_1] = False

                    if self.flags[FlagIndex.WVF_PARADE]:
                        process_waveform_parade(self.rgb1_8,self.wvf1_8)
                    else:
                        process_waveform_stacked(self.rgb1_8,self.wvf1_8)

                    self.flags[FlagIndex.READY_WVF0]=False
                    self.flags[FlagIndex.READY_WVF1]=True

                time.sleep(0.001)

        except KeyboardInterrupt:
            log(LogTag.WFM, "Interrupted by user")
        except Exception as e:
            log(LogTag.WFM, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()
        finally:
            log(LogTag.WFM, "Stopping...")


def main():
    """Main function for Waveform module"""
    try:
        waveform = DisplayWaveform()
        waveform.process()
    except Exception as e:
        log(LogTag.WFM, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.WFM, "Shutting down...")
        cleanup_manager()
        log(LogTag.WFM, "Cleanup complete")


if __name__ == "__main__":
    main()