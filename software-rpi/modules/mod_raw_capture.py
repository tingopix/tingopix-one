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
Raw capture / stacking processor

Owns the stacked-capture pipeline that produces the final archival frame.
On a capture request, sets per-channel LED and exposure, then chases fresh
camera frames from CAP's raw0_16/raw1_16 buffers into a ring, and runs the
batched Numba accumulator (stack_sum_slots) to sum them into raw_capture.
Runs RGB (separate per-channel exposure) or all-channels (single exposure)
mode, embeds film-position and gain metadata into the last column of
raw_capture, and signals completion. First module to compile.

Shared memory:
    Reads:  MOD_READY_ALL, SHUTDOWN_REQUESTED, RQST_CAPTURE, MODE_CAPTURE_RGB,
            RAW_FRAME_CLK, READY_LED_G/R/B/RGB/DET, BUFFER_COUNTER,
            FRAME_COUNTER, EXPOSURE, EXP_DET/RGB/G/R/B, RAW0_EXPOSURE,
            RAW1_EXPOSURE, STACKING_SETTING, FILM_LOCATION_VERTICAL/HORIZONTAL,
            SPROCKET_MARKER, CROP_HEIGHT, R_DIG_GAIN, B_DIG_GAIN,
            raw0_16, raw1_16, raw_buffer_16, raw_stack_16
    Writes: MOD_READY_RAWCAPTURE, READY_CAPTURE, RQST_CAPTURE (clears),
            RQST_LED_G/R/B/RGB/DET, READY_LED_G/R/B/RGB/DET (clears),
            EXPOSURE, CAPTURE_DEPTH_G/R/B, raw_capture_16, raw_stack_16

Interacts with:
    mod_capture (RAW_FRAME_CLK, BUFFER_COUNTER, EXPOSURE, raw0/1_16),
    mod_serial (RQST_LED_*/READY_LED_*), mod_file_saver (READY_CAPTURE,
    embedded metadata in raw_capture), shm_manager (arrays)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, Uint16Index, Uint32Index, FloatIndex
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

import time
import numpy as np
from numba import njit

@njit (cache=True)
def capture_color(buffer_raw, capture_raw, color=3, clear_capture=False, overwrite_capture=True):
    if clear_capture:
        capture_raw[:] = 0
    else:
        if overwrite_capture:
            # Overwrite color
            if color == 3:
                # Write all colors
                capture_raw[:] = buffer_raw

            if color == 0:
                # Write green
                capture_raw[::2, ::2] = buffer_raw[::2, ::2]
                capture_raw[1::2, 1::2] = buffer_raw[1::2, 1::2]

            if color == 1:
                # Write red
                capture_raw[1::2, ::2] = buffer_raw[1::2, ::2]

            if color == 2:
                # Write blue
                capture_raw[::2, 1::2] = buffer_raw[::2, 1::2]
        else:
            # Add color capture
            if color == 3:
                # Write all colors
                capture_raw[:] += buffer_raw

            if color == 0:
                # Write green
                capture_raw[::2, ::2] += buffer_raw[::2, ::2]
                capture_raw[1::2, 1::2] += buffer_raw[1::2, 1::2]

            if color == 1:
                # Write red
                capture_raw[1::2, ::2] += buffer_raw[1::2, ::2]

            if color == 2:
                # Write blue
                capture_raw[::2, 1::2] += buffer_raw[::2, 1::2]


@njit (cache=True)
def stack_sum_slots(ring, capture_raw, count, color):
    """
    Sum the first `count` slots of `ring` into `capture_raw` for pixels
    belonging to the given Bayer `color` channel.

    Uses uint16 accumulator (variable `s`) — safe at depth ≤ 16 with 12-bit
    source data (max sum = 16 * 4095 = 65520 < uint16 max 65535). No
    saturation check.

    NOTE: If RAW_STACK_DEPTH is ever raised above 16, or source bit depth
    above 12, this function must revisit overflow handling.

    ring         : uint16[N, H, W]  ring of captured frames
    capture_raw  : uint16[H, W]     destination (overwritten for active color)
    count        : int              number of valid slots in ring (1..N)
    color        : int              0=green, 1=red, 2=blue, 3=all
    """
    H, W = capture_raw.shape

    if color == 3:
        # All channels — sum every pixel
        for i in range(H):
            for j in range(W):
                s = ring[0, i, j]
                for k in range(1, count):
                    s += ring[k, i, j]
                capture_raw[i, j] = s

    elif color == 0:
        # Green — Gr (even/even) and Gb (odd/odd)
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                s = ring[0, i, j]
                for k in range(1, count):
                    s += ring[k, i, j]
                capture_raw[i, j] = s
        for i in range(1, H, 2):
            for j in range(1, W, 2):
                s = ring[0, i, j]
                for k in range(1, count):
                    s += ring[k, i, j]
                capture_raw[i, j] = s

    elif color == 1:
        # Red — odd row, even col
        for i in range(1, H, 2):
            for j in range(0, W, 2):
                s = ring[0, i, j]
                for k in range(1, count):
                    s += ring[k, i, j]
                capture_raw[i, j] = s

    elif color == 2:
        # Blue — even row, odd col
        for i in range(0, H, 2):
            for j in range(1, W, 2):
                s = ring[0, i, j]
                for k in range(1, count):
                    s += ring[k, i, j]
                capture_raw[i, j] = s


class RawCaptureProcessor:
    def __init__(self):
        log(LogTag.RCAP, "Initializing...")
        
        # Get arrays using config keys
        self.flags = get_array("flags")
        self.raw_buffer = get_array("shm_raw_buffer_16")
        self.raw_capture = get_array("shm_raw_capture_16")
        self.raw0_16 = get_array("shm_raw0_16")
        self.raw1_16 = get_array("shm_raw1_16")
        self.raw_stack = get_array("shm_raw_stack_16")  # shape (RAW_STACK_DEPTH, H, W)
        self.ring_depth = self.raw_stack.shape[0]
        self.sm_uint16var = get_array("shm_uint16_var")
        self.sm_uint32var = get_array("shm_uint32_var")
        self.sm_floatvar  = get_array("shm_float_var")


        if self.flags is None or self.raw_buffer is None or self.raw_capture is None or self.sm_uint32var is None or self.sm_floatvar is None:
            raise RuntimeError("Raw Capture Processor Module: Failed to get shared arrays")
        
        log(LogTag.RCAP, "Connected to shared memory")

        self.BIT_DEPTH_LOOK_UP = (12, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16)

        #First Module to compile
        # No wait for previous module

        log_compile_start(LogTag.RCAP, "capture function")
        compile_start = time.perf_counter()
        # Warm-up: trivial clear path
        capture_color(self.raw_buffer, self.raw_capture, 3, True, False)
        # Warm-up: real capture path (overwrite=True, color=3) — the main runtime branch
        capture_color(self.raw_buffer, self.raw_capture, 3, False, True)
        # Warm-up: accumulate path (overwrite=False) — used for stacking
        capture_color(self.raw_buffer, self.raw_capture, 3, False, False)
        # Warm-up: ring slot writer (raw0/1 → ring slot)
        capture_color(self.raw0_16, self.raw_stack[0], 3, False, True)
        # Warm-up: ring slot accumulator (ring slot → capture)
        capture_color(self.raw_stack[0], self.raw_capture, 3, False, False)
        # Warm-up: batched stack accumulator (all four color modes).
        # Use count=2 so the inner `for k in range(1, count)` loop actually
        # executes — count=1 would short-circuit and leave the inner add path
        # uncompiled until first runtime call.
        stack_sum_slots(self.raw_stack, self.raw_capture, 2, 3)
        stack_sum_slots(self.raw_stack, self.raw_capture, 2, 0)
        stack_sum_slots(self.raw_stack, self.raw_capture, 2, 1)
        stack_sum_slots(self.raw_stack, self.raw_capture, 2, 2)
        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.RCAP, compile_time)
        log(LogTag.RCAP, "Ready")

        #Raised finished compiling
        self.flags[FlagIndex.MOD_READY_RAWCAPTURE] = True

        #Wait until last module finished compiling
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)


    def wait_for_buffer_updates(self, num_frames, timeout=10.0):
        """
        Wait for a specific number of buffer updates with timeout protection.
        
        This waits for the counter to CHANGE num_frames times, not reach a specific value.
        This handles exposure changes where counter might not increment sequentially.
        
        Args:
            num_frames: Number of buffer updates to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if successful, False if timeout
        """
        start_time = time.perf_counter()
        updates_seen = 0
        last_counter = self.sm_uint32var[Uint32Index.BUFFER_COUNTER]
        
        while updates_seen < num_frames:
            current_counter = self.sm_uint32var[Uint32Index.BUFFER_COUNTER]
            
            # Check if counter changed (ANY change counts, not just +1)
            if current_counter != last_counter:
                updates_seen += 1
                last_counter = current_counter
            
            if time.perf_counter() - start_time > timeout:
                log(LogTag.RCAP, f"Buffer wait timeout after {timeout}s — waited {num_frames} updates, got {updates_seen}", level="WARN")
                log(LogTag.RCAP, f"  Counter stuck at {current_counter} | lock={self.flags[FlagIndex.LOCK_BUFFER_16_WRITE]} | exp target={self.sm_uint32var[Uint32Index.EXPOSURE]}µs raw0={self.sm_uint32var[Uint32Index.RAW0_EXPOSURE]}µs raw1={self.sm_uint32var[Uint32Index.RAW1_EXPOSURE]}µs", level="WARN")
                return False
            
            time.sleep(0.001)  # Short sleep to avoid busy-wait
        
        return True

    def wait_for_exposure_stable(self, timeout=10.0, tolerance_us=100):
        """
        Wait for exposure to stabilize after a change.
        Monitors both RAW0 and RAW1 exposure to match the set exposure.
        
        During this wait, BUFFER_COUNTER will NOT increment because
        frame1_exposure_confirmed will be False in mod_capture.
        
        Args:
            timeout: Maximum time to wait in seconds
            tolerance_us: Acceptable difference in microseconds (default 100µs)
        
        Returns:
            bool: True if exposure stabilized, False if timeout
        """
        target_exposure = int(self.sm_uint32var[Uint32Index.EXPOSURE])
        start_time = time.perf_counter()
        
        stable_frames = 0
        required_stable = 3  # Require 3 consecutive matching frames for reliability
        
        while stable_frames < required_stable:
            if time.perf_counter() - start_time > timeout:
                log(LogTag.RCAP, f"Exposure stabilization timeout after {timeout}s — target={target_exposure}µs raw0={int(self.sm_uint32var[Uint32Index.RAW0_EXPOSURE])}µs raw1={int(self.sm_uint32var[Uint32Index.RAW1_EXPOSURE])}µs (tol={tolerance_us}µs)", level="WARN")
                return False
            
            # Check if both frames have exposure within tolerance
            # Convert to signed int to avoid overflow with uint32
            raw0_exp = int(self.sm_uint32var[Uint32Index.RAW0_EXPOSURE])
            raw1_exp = int(self.sm_uint32var[Uint32Index.RAW1_EXPOSURE])
            
            raw0_match = abs(raw0_exp - target_exposure) <= tolerance_us
            raw1_match = abs(raw1_exp - target_exposure) <= tolerance_us
            
            if raw0_match and raw1_match:
                stable_frames += 1
            else:
                stable_frames = 0  # Reset if mismatch
            
            time.sleep(0.005)  # Check every 5ms
        
        return True

    def set_exposure_and_wait(self, exposure_us, stabilization_frames=6):
        """
        Set exposure and wait for it to stabilize.
        
        During exposure change, the capture module STOPS incrementing BUFFER_COUNTER
        for 4-6 frames while confirming the new exposure. We need to account for this.
        
        Args:
            exposure_us: Exposure time in microseconds
            stabilization_frames: Number of CONFIRMED frames to wait after exposure is stable
            
        Returns:
            bool: True if successful, False if failed
        """
        self.sm_uint32var[Uint32Index.EXPOSURE] = exposure_us
        
        # Wait for exposure to be confirmed in the capture module
        # During this time, BUFFER_COUNTER will NOT increment!
        if not self.wait_for_exposure_stable(timeout=10.0):
            return False
        
        # NOW wait for buffer updates - counter should increment normally again
        return self.wait_for_buffer_updates(stabilization_frames, timeout=10.0)

    def capture_with_stacking(self, color, count, exposure_us=None):
        """
        Capture a specific color with ring-buffer stacking.

        CAP keeps running at full speed (live monitor unaffected). RCAP reads
        raw0_16 / raw1_16 directly into a small ring buffer, and accumulates
        each slot into raw_capture one slot behind the writer. CAP and RCAP
        never touch the same physical buffer, so no LOCK_BUFFER_16_WRITE is
        needed during the stack.
        
        Total frames stacked = count + 1 (first frame overwrites, remaining
        `count` frames accumulate). Camera-bound at ~125 ms/frame on Pi 4.
        
        Args:
            color: Color channel (0=green, 1=red, 2=blue, 3=all)
            count: Number of additional frames to stack on top of the first
            exposure_us: Exposure to use (None to keep current)
        
        Returns:
            bool: True if successful, False on timeout
        """
        # Set exposure if specified (waits for stabilization)
        if exposure_us is not None:
            current_exposure = self.sm_uint32var[Uint32Index.EXPOSURE]
            
            if exposure_us != current_exposure:
                log(LogTag.RCAP, f"Exposure change: {current_exposure} → {exposure_us}µs")
                if not self.set_exposure_and_wait(exposure_us):
                    log(LogTag.RCAP, f"Failed to set exposure for color {color}", level="WARN")
                    return False
            else:
                self.sm_uint32var[Uint32Index.EXPOSURE] = exposure_us
        
        # LED already settled in serial module before READY flag was set.
        # Wait one frame for the camera to deliver post-LED-settling data.
        if not self.wait_for_buffer_updates(1, timeout=10.0):
            log(LogTag.RCAP, f"Timeout waiting for LED settling frame for color {color}", level="WARN")
            return False

        total_frames = int(count) + 1
        depth = self.ring_depth

        if total_frames > depth:
            log(LogTag.RCAP,
                f"Stack count {total_frames} exceeds ring depth {depth} — "
                f"clamping to {depth}", level="WARN")
            total_frames = depth

        t_loop_start = time.perf_counter()

        # Phase 1: write `total_frames` fresh raw frames into ring slots.
        # No accumulation in this phase — RCAP is camera-bound, idle time per
        # iteration is "wasted" (could be ~85ms after a 40ms write within the
        # 125ms camera period). That's the cost of two-phase; in exchange we
        # get a single batched accumulator pass with much better cache reuse.
        for stack_idx in range(total_frames):
            # Wait for next fresh camera frame (BUFFER_COUNTER tick from CAP)
            if not self.wait_for_buffer_updates(1, timeout=10.0):
                log(LogTag.RCAP,
                    f"Timeout on stack frame {stack_idx+1}/{total_frames} "
                    f"for color {color}", level="WARN")
                return False

            # After BUFFER_COUNTER tick, RAW_FRAME_CLK indicates which raw
            # buffer CAP just finished writing.
            #   RAW_FRAME_CLK == True  → fresh frame is in raw1_16
            #   RAW_FRAME_CLK == False → fresh frame is in raw0_16
            src = self.raw1_16 if self.flags[FlagIndex.RAW_FRAME_CLK] else self.raw0_16

            # Writer: full-frame contiguous copy into next ring slot. Color
            # masking is deferred to Phase 2's batched accumulator.
            capture_color(src, self.raw_stack[stack_idx], 3, False, True)

        # Phase 2: batched accumulate — sum the first `total_frames` slots
        # into raw_capture for the active color channel. Single pass, single
        # write to raw_capture per pixel. uint16 accumulator is safe: max
        # stack 16 × 12-bit (4095) = 65520 < 65535.
        stack_sum_slots(self.raw_stack, self.raw_capture, total_frames, color)

        t_total_ms = (time.perf_counter() - t_loop_start) * 1000
        log(LogTag.RCAP, f"Stack color={color} N={total_frames}: {t_total_ms:.0f}ms")

        return True

    def process(self):
        """Main processing loop"""
        log(LogTag.RCAP, "Starting capture loop...")
        
        try:
            # Initialize settings

            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                # Wait for capture request from GUI
                if not self.flags[FlagIndex.RQST_CAPTURE]:
                    time.sleep(0.01)
                    continue
                
                # Clear the request flag
                self.flags[FlagIndex.RQST_CAPTURE] = False
                self.flags[FlagIndex.READY_CAPTURE] = False
                
                log(LogTag.RCAP, "Starting capture sequence...")

                ### NEED TO PRESERVE FRAME POSITION BEFORE 
                film_position = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
                film_edge = self.sm_uint16var[Uint16Index.FILM_LOCATION_HORIZONTAL]
                marker_position = self.sm_uint16var[Uint16Index.SPROCKET_MARKER]
                crop_height = self.sm_uint16var[Uint16Index.CROP_HEIGHT]

                log(LogTag.RCAP, f"Capture params — V:{film_position} H:{film_edge} marker:{marker_position} crop:{crop_height}")                 

                # Check for Control Panel Current Info
                current_stacking = self.sm_uint16var[Uint16Index.STACKING_SETTING] - 1

                exposure_det = self.sm_uint32var[Uint32Index.EXP_DET] 
                exposure_rgb = self.sm_uint32var[Uint32Index.EXP_RGB]
                exposure_g = self.sm_uint32var[Uint32Index.EXP_G]
                exposure_r = self.sm_uint32var[Uint32Index.EXP_R]
                exposure_b = self.sm_uint32var[Uint32Index.EXP_B]

                
                count_capture_all = self.sm_uint16var[Uint16Index.STACKING_SETTING] - 1
                count_capture_g = count_capture_all
                count_capture_r = count_capture_all
                count_capture_b = count_capture_all

                exposure_setting = [exposure_g, exposure_r, exposure_b, exposure_rgb]

                flag_index_led_request = [FlagIndex.RQST_LED_G, FlagIndex.RQST_LED_R, FlagIndex.RQST_LED_B, FlagIndex.RQST_LED_RGB, FlagIndex.RQST_LED_DET] 
                flag_index_led_ready = [FlagIndex.READY_LED_G, FlagIndex.READY_LED_R, FlagIndex.READY_LED_B, FlagIndex.READY_LED_RGB, FlagIndex.READY_LED_DET] 

                count_setting = [count_capture_g, count_capture_r, count_capture_b,count_capture_g] # index 3 correcsponds to RGB

                self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G] = self.BIT_DEPTH_LOOK_UP[count_setting[0]]
                self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R] = self.BIT_DEPTH_LOOK_UP[count_setting[1]]
                self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B] = self.BIT_DEPTH_LOOK_UP[count_setting[2]]


                if current_stacking != count_capture_all:
                    for count_ndx in range(3):
                        count_capture_all = current_stacking
                        count_setting[count_ndx] = current_stacking
                    self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G] = self.BIT_DEPTH_LOOK_UP[count_setting[0]]
                    self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R] = self.BIT_DEPTH_LOOK_UP[count_setting[1]]
                    self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B] = self.BIT_DEPTH_LOOK_UP[count_setting[2]]

                capture_start_frame = self.sm_uint32var[Uint32Index.FRAME_COUNTER]
                capture_mode = self.flags[FlagIndex.MODE_CAPTURE_RGB]

                # Clear capture buffer
                capture_color(self.raw_buffer, self.raw_capture, 3, True, False)

                if capture_mode:
                    # RGB CAPTURE -> SEPARATE EXPOSURE
                    log(LogTag.RCAP, "RGB capture mode — separate exposures per channel")

                    for color_idx in range(3):
                        color_name = ['Green', 'Red', 'Blue'][color_idx]

                        #Direct request to Serial Module for light change
                        self.flags[flag_index_led_ready[color_idx]] = False
                        self.flags[flag_index_led_request[color_idx]] = True
                        led_ready = False
                        time.sleep(0.1)
                        while not led_ready:
                            led_ready = self.flags[flag_index_led_ready[color_idx]]
                            time.sleep(0.05)

                        if not self.capture_with_stacking(
                            color=color_idx,
                            count=count_setting[color_idx],
                            exposure_us=exposure_setting[color_idx]
                        ):
                            log(LogTag.RCAP, f"Failed to capture {color_name} channel", level="WARN")
                            break
                else:
                    # ALL CHANNELS -> SAME EXPOSURE
                    log(LogTag.RCAP, "All-channels capture mode — single exposure")

                    color_idx=3 # Set LED for RGB
                    #Direct request to Serial Module for light change
                    self.flags[flag_index_led_ready[color_idx]] = False
                    self.flags[flag_index_led_request[color_idx]] = True
                    led_ready = False
                    time.sleep(0.1)
                    while not led_ready:
                        led_ready = self.flags[flag_index_led_ready[color_idx]]
                        time.sleep(0.05)

                    if not self.capture_with_stacking(
                        color=3,  # All colors
                        count=count_setting[color_idx],
                        exposure_us=exposure_setting[color_idx]
                    ):
                        log(LogTag.RCAP, "Failed to capture all channels", level="WARN")

                # Restore default exposure
                current_exposure = self.sm_uint32var[Uint32Index.EXPOSURE]
                if exposure_det != current_exposure:
                    self.sm_uint32var[Uint32Index.EXPOSURE] = exposure_det
                    self.wait_for_buffer_updates(5, timeout=10.0)
                else:
                    self.sm_uint32var[Uint32Index.EXPOSURE] = exposure_det
                   

                color_idx=4 # Set LED for Detector                    
                #Direct request to Serial Module for light change
                self.flags[flag_index_led_ready[color_idx]] = False
                self.flags[flag_index_led_request[color_idx]] = True
                led_ready = False
                time.sleep(0.1)
                while not led_ready:
                    led_ready = self.flags[flag_index_led_ready[color_idx]]
                    time.sleep(0.05)


                frames_used = self.sm_uint32var[Uint32Index.FRAME_COUNTER] - capture_start_frame
                log(LogTag.RCAP, f"Capture complete — {frames_used} frames used")

                #EMBED FILM LOCATION TO RAW CAPTURE ARRAY

                # Calculate horizontal AOI left edge (rightmost 128 pixels)
                aoi_width = 128  # Width of horizontal edge detection AOI
                frame_width_binned = 2032
                aoi_left = frame_width_binned - aoi_width  # = 1904

                # Convert from binned to absolute full-resolution coordinates
                # Guard sentinel value 0xFFFF — pass through unchanged, do not scale
                self.raw_capture[0][-1] = film_position * 2 if film_position < 0xFFFF else 0xFFFF      # vertical position
                _h_abs = int(aoi_left) + int(film_edge)
                self.raw_capture[1][-1] = _h_abs * 2 if film_edge < 0xFFFF and _h_abs * 2 <= 0xFFFF else 0xFFFF  # horizontal edge (absolute)
                self.raw_capture[2][-1] = marker_position * 2    # marker reference
                self.raw_capture[3][-1] = crop_height * 2        # crop height

                _embedded_v = self.raw_capture[0][-1]
                _embedded_h = self.raw_capture[1][-1]
                _no_detection = (_embedded_v == 0xFFFF or _embedded_h == 0xFFFF)

                if capture_mode: # if capture mode is RGB
                    self.raw_capture[4][-1] = 0x0000
                    self.raw_capture[5][-1] = 1000  # gain_r = 1.000
                    self.raw_capture[6][-1] = 1000  # gain_b = 1.000
                    log(LogTag.RCAP, f"Embedded — V:{_embedded_v} H:{_embedded_h} marker:{marker_position * 2} crop:{crop_height * 2} gain_r:1.000 gain_b:1.000" + (" [NO DETECTION]" if _no_detection else ""), level="WARN" if _no_detection else "INFO")
                else:
                    self.raw_capture[4][-1] = 0x0001
                    self.raw_capture[5][-1] = int(self.sm_floatvar[FloatIndex.R_DIG_GAIN] * 1000)
                    self.raw_capture[6][-1] = int(self.sm_floatvar[FloatIndex.B_DIG_GAIN] * 1000)
                    log(LogTag.RCAP, f"Embedded — V:{_embedded_v} H:{_embedded_h} marker:{marker_position * 2} crop:{crop_height * 2} gain_r:{self.sm_floatvar[FloatIndex.R_DIG_GAIN]:.3f} gain_b:{self.sm_floatvar[FloatIndex.B_DIG_GAIN]:.3f}" + (" [NO DETECTION]" if _no_detection else ""), level="WARN" if _no_detection else "INFO")

                # Signal capture complete
                self.flags[FlagIndex.READY_CAPTURE] = True

        except KeyboardInterrupt:
            log(LogTag.RCAP, "Interrupted by user")
        except Exception as e:
            log(LogTag.RCAP, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()
        finally:
            log(LogTag.RCAP, "Stopping...")


def main():
    """Main function for capture module"""
    processor = None
    try:
        processor = RawCaptureProcessor()
        processor.process()
    except Exception as e:
        log(LogTag.RCAP, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.RCAP, "Shutting down...")
        cleanup_manager()
        log(LogTag.RCAP, "Cleanup complete")


if __name__ == "__main__":
    main()