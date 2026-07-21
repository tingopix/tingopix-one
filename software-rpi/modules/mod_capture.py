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
Capture processor

Drives the IMX477 camera and owns the raw-capture front of the pipeline.
Ping-pongs full-res 12-bit raw frames into two buffers (raw0_16/raw1_16),
handles exposure-change requests with per-frame confirmation, copies confirmed
frames into the shared raw buffer, and raises debayer requests for the sprocket
and debayer processors. Runs on its own core and starts only after all modules
report compiled.

Shared memory:
    Reads:  MOD_READY_GUI_AOI, SHUTDOWN_REQUESTED, RAW_FRAME_CLK,
            LOCK_BUFFER_16_WRITE, EXPOSURE, EXP_RGB, FRAME_COUNTER,
            BUFFER_COUNTER
    Writes: RAW_FRAME_CLK, MOD_READY_ALL, READY_BUFFER_16, RQST_DBY_B0,
            RQST_DBY_B1, RQST_DBY_RGB0, RQST_DBY_RGB1, FRAME_COUNTER,
            BUFFER_COUNTER, RAW0_EXPOSURE, RAW1_EXPOSURE, CAPTURE_DEPTH_R/G/B,
            FT_CAPTURE_TIME, raw0_16, raw1_16, raw_buffer_16
    Writes (via auto_load_defaults): EXPOSURE, EXP_DET, EXP_RGB, light defaults

Interacts with:
    mod_sprocket (RQST_DBY_B0/B1), mod_debayer (RQST_DBY_RGB0/RGB1),
    mod_exp_defaults (auto_load_defaults), shm_manager (arrays)
"""

from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, FloatIndex, Uint16Index, Uint32Index
from modules.mod_exp_defaults import auto_load_defaults
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

# HQ Camera
from picamera2 import Picamera2
from libcamera import controls,Transform

import time
import numpy as np
from numba import njit

@njit (cache=True)
def buffer_transfer(raw_frame, buffer_frame):
    buffer_frame[:] = raw_frame


class CaptureProcessor:
    def __init__(self):

        log(LogTag.CAP, "Initializing...")
        
        # Get arrays using config keys
        self.flags = get_array("flags")
        self.raw0_16 = get_array("shm_raw0_16")
        self.raw1_16 = get_array("shm_raw1_16")
        self.raw_buffer_16 = get_array("shm_raw_buffer_16")
        self.sm_floatvar = get_array("shm_float_var")
        self.sm_uint16var = get_array("shm_uint16_var")
        self.sm_uint32var = get_array("shm_uint32_var")
        
        if (self.flags is None or self.raw0_16 is None or self.raw1_16 is None or 
            self.raw_buffer_16 is None or self.sm_floatvar is None or self.sm_uint32var is None):
            raise RuntimeError("Module Capture: Failed to get shared arrays")
        
        log(LogTag.CAP, "Connected to shared memory")

        # Initialize exposure and light defaults from centralized module
        # Safe to call even if GUI already initialized - values will be consistent
        auto_load_defaults(self.sm_uint32var, self.sm_uint16var, self.sm_floatvar)

        # Initialize capture-specific values (bit depth)
        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R] = 12
        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G] = 12
        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B] = 12

        global_shape_main = (height, width) = 4064,3040
        global_shape_sensor = (width,height) = 4064,3040


        #Wait for all modules to compile before starting camera

        while not self.flags[FlagIndex.MOD_READY_GUI_AOI]:
            time.sleep(0.5)  # delay Waiting for All modules to compile

        log_compile_start(LogTag.CAP, "buffer transfer function")
        compile_start = time.perf_counter()
        buffer_transfer(self.raw0_16, self.raw_buffer_16)
        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.CAP, compile_time)


        log(LogTag.CAP, "Initializing camera (IMX477)...")
        # Stock libcamera tuning; resolved from the system tuning directory
        # (/usr/share/libcamera/ipa/rpi/...), not from the project.
        tune = Picamera2.load_tuning_file("imx477_scientific.json")
        # To experiment with a modified tuning, point at your own file instead:
        #   tune = Picamera2.load_tuning_file("/full/path/to/my_tuning.json")

        self.pihqcam = Picamera2(tuning=tune)
        self.cam_controls = {
            "AwbEnable": False,
            "AeEnable": False,
            "FrameRate": 30.0, #this will not affect the frame rate of captures via capture array "raw", which is a blocking method.
            "AnalogueGain": 1.0,
            "Sharpness": 1.0,
            "Saturation":1.0,
            "ColourGains": (2.8, 1.9),# (R,B)
            "ExposureTime": self.sm_uint32var[Uint32Index.EXP_RGB],
  
            "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off,
            "FrameDurationLimits": (100, 98000)
        }

        self.pihqcam.create_preview_configuration(queue=False)

        self.pihqcam.preview_configuration.main.size = global_shape_main #this should be 4056, 3040
        self.pihqcam.preview_configuration.main.format = "RGB888"
        self.pihqcam.preview_configuration.transform=Transform(hflip=True)

        self.pihqcam.preview_configuration.raw.size = global_shape_sensor
        self.pihqcam.preview_configuration.raw.format = "SRGGB12"

        self.pihqcam.preview_configuration.buffer_count = 4

        self.pihqcam.configure(self.pihqcam.preview_configuration)
        log(LogTag.CAP, "Camera configured — preview mode, RGB888 + SRGGB12 raw")

        self.pihqcam.set_controls(self.cam_controls)
        self.pihqcam.start()
        log(LogTag.CAP, "Camera started")

        self.buffer_counter = self.sm_uint32var[Uint32Index.BUFFER_COUNTER]

        self.flags[FlagIndex.MOD_READY_ALL]= True


        
    def process(self):
        """Main processing loop"""
        log(LogTag.CAP, "Starting capture loop...")
        
        counter = 0
        start_time = time.perf_counter()
        
        self.current_exposure = self.sm_uint32var[Uint32Index.EXPOSURE]
        self.frame0_exposure_confirmed = True
        self.frame1_exposure_confirmed = True

        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                self.exposure_setting = self.sm_uint32var[Uint32Index.EXPOSURE]
                
                # Detect exposure change request
                if self.current_exposure != self.exposure_setting:
                    self.current_exposure = self.exposure_setting
                    self.pihqcam.set_controls({"ExposureTime": self.current_exposure})
                    # Both frames need to confirm the new exposure
                    self.frame0_exposure_confirmed = False
                    self.frame1_exposure_confirmed = False

                #Alternate Capture Frame
                raw_frame_clk = not self.flags[FlagIndex.RAW_FRAME_CLK] 
                self.flags[FlagIndex.RAW_FRAME_CLK] = raw_frame_clk

                frame_counter = self.sm_uint32var[Uint32Index.FRAME_COUNTER]

                if (self.flags[FlagIndex.RAW_FRAME_CLK]):
                    self.raw1_16[:] = (self.pihqcam.capture_array("raw").view(np.uint16))
                    self.sm_uint32var[Uint32Index.FRAME_COUNTER] = frame_counter + 1
                    
                    # Only read metadata if frame1 needs to confirm exposure
                    if not self.frame1_exposure_confirmed:
                        metadata = self.pihqcam.capture_metadata()
                        self.exposure_us = metadata.get('ExposureTime', 0)
                        self.sm_uint32var[Uint32Index.RAW1_EXPOSURE] = self.exposure_us
                        # Use tolerance instead of exact match (camera may round exposure values)
                        self.frame1_exposure_confirmed = (abs(int(self.exposure_setting) - int(self.exposure_us)) <= 100)
                    
                    # LOCK_BUFFER_16_WRITE will while storing a color (or colors) from the buffer.
                    if (self.frame1_exposure_confirmed and not self.flags[FlagIndex.LOCK_BUFFER_16_WRITE]):
                        self.flags[FlagIndex.READY_BUFFER_16] = False
                        self.buffer_counter = self.sm_uint32var[Uint32Index.BUFFER_COUNTER] + 1
                        buffer_transfer(self.raw1_16, self.raw_buffer_16)
                        self.sm_uint32var[Uint32Index.BUFFER_COUNTER] = self.buffer_counter
                        self.flags[FlagIndex.READY_BUFFER_16] = True


                    #Processing Request to Sprocket Processor
                    self.flags[FlagIndex.RQST_DBY_B0] = False    # Request Debayering of B0
                    self.flags[FlagIndex.RQST_DBY_B1] = True   # Request Debayering of B1

                    #Processing Request to Debayer Processor
                    self.flags[FlagIndex.RQST_DBY_RGB0] = False   # Request Debayering of RGB0
                    self.flags[FlagIndex.RQST_DBY_RGB1] = True   # Request Debayering of RGB1

                else:
                    self.raw0_16[:] = (self.pihqcam.capture_array("raw").view(np.uint16))
                    self.sm_uint32var[Uint32Index.FRAME_COUNTER] = frame_counter + 1
                    
                    # Only read metadata if frame0 needs to confirm exposure
                    if not self.frame0_exposure_confirmed:
                        metadata = self.pihqcam.capture_metadata()
                        self.exposure_us = metadata.get('ExposureTime', 0)
                        self.sm_uint32var[Uint32Index.RAW0_EXPOSURE] = self.exposure_us
                        # Use tolerance instead of exact match (camera may round exposure values)
                        self.frame0_exposure_confirmed = (abs(int(self.exposure_setting) - int(self.exposure_us)) <= 100)

                    # LOCK_BUFFER_16_WRITE will while storing a color (or colors) from the buffer.
                    if (self.frame0_exposure_confirmed and not self.flags[FlagIndex.LOCK_BUFFER_16_WRITE]):
                        self.flags[FlagIndex.READY_BUFFER_16] = False
                        self.buffer_counter = self.sm_uint32var[Uint32Index.BUFFER_COUNTER] + 1
                        buffer_transfer(self.raw0_16, self.raw_buffer_16)
                        self.sm_uint32var[Uint32Index.BUFFER_COUNTER] = self.buffer_counter
                        self.flags[FlagIndex.READY_BUFFER_16] = True

                    #Processing Request to Sprocket Processor
                    self.flags[FlagIndex.RQST_DBY_B1] = False   # Request Debayering of B1
                    self.flags[FlagIndex.RQST_DBY_B0] = True    # Request Debayering of B0

                    #Processing Request to Debayer Processor
                    self.flags[FlagIndex.RQST_DBY_RGB1] = False   # Request Debayering of RGB1
                    self.flags[FlagIndex.RQST_DBY_RGB0] = True   # Request Debayering of RGB0

              
                if counter % 100 == 0:
                    end_time = time.perf_counter()
                    self.sm_floatvar[FloatIndex.FT_CAPTURE_TIME] = 100/(end_time - start_time)
                    start_time = time.perf_counter()
                
                counter += 1

        except KeyboardInterrupt:
            log(LogTag.CAP, "Interrupted by user")
        except Exception as e:
            log(LogTag.CAP, f"Error — {e}", level="ERR")
        finally:
            log(LogTag.CAP, "Stopping camera...")
            try:
                self.pihqcam.stop()
                log(LogTag.CAP, "Camera stopped successfully")
            except Exception as e:
                log(LogTag.CAP, f"Error stopping camera — {e}", level="ERR")

def main():
    """Main function for capture module"""
    processor = None
    try:
        processor = CaptureProcessor()
        processor.process()
    except Exception as e:
        log(LogTag.CAP, f"Fatal error — {e}", level="ERR")
    finally:
        log(LogTag.CAP, "Shutting down...")
        if processor and hasattr(processor, 'pihqcam'):
            try:
                processor.pihqcam.stop()
                log(LogTag.CAP, "Camera stopped in cleanup")
            except:
                pass
        cleanup_manager()
        log(LogTag.CAP, "Cleanup complete")