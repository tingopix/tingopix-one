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
Display debayer (8x8 / focus)

Debayers the live raw monitor stream into the small 380×508 RGB preview and
drives the focus-assist path. Bins 8×8 for the wide view or 2×2 (center crop)
for focus mode, optionally applies BLC + R/B gain + CCM color processing, and
computes a per-channel Tenengrad focus score with on-screen focus bars. Reads
the raw ping-pong buffers or the captured buffer depending on show-buffer
state, and hands each debayered frame to the waveform processor.

Shared memory:
    Reads:  MOD_READY_RAWCAPTURE, MOD_READY_ALL, SHUTDOWN_REQUESTED,
            RQST_DBY_RGB0, RQST_DBY_RGB1, RGB_8_FOCUS, RGB_8_FOCUS_INDICATOR,
            RGB_8_COLOR_PROCESSING, RGB_8_SHOW_BUFFER, RGB_8_INVERT,
            MODE_CAPTURE_RGB, CAPTURE_DEPTH_R/G/B, R_DIG_GAIN, B_DIG_GAIN,
            raw0_16, raw1_16, raw_capture_16
    Writes: MOD_READY_DEBAYER, RQST_DBY_RGB0/RGB1 (clears), RQST_WVF_0,
            RQST_WVF_1, READY_RGB0, READY_RGB1, CAPTURE_DEPTH_R/G/B,
            R_DIG_GAIN, B_DIG_GAIN (seed defaults), rgb0_8, rgb1_8

Interacts with:
    mod_capture (RQST_DBY_RGB0/RGB1), mod_raw_capture (MOD_READY_RAWCAPTURE,
    raw_capture_16), mod_waveform (RQST_WVF_0/1, READY_RGB0/1),
    mod_debayer_with_blc_ccm (get_ccm_matrix),
    shm_manager (arrays)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, Uint16Index, FloatIndex
from tpx_config import DEFAULT_R_DIG_GAIN, DEFAULT_B_DIG_GAIN
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

import time
from numba import njit, prange
import numpy as np
from modules.mod_debayer_with_blc_ccm import get_ccm_matrix
from tpx_config import CCM_COLOR_TEMP



@njit(parallel=True, cache=True)
def debayer_with_focus(raw_frame, r_bit_depth, g_bit_depth, b_bit_depth, rgb_frame, focus_mode, draw_focus_bar=True, apply_color=False, apply_gain=False, ccm=None, gain_r=np.float32(1.0), gain_b=np.float32(1.0)):
    """
    Debayer with two modes and focus metric calculation.
    
    Args:
        raw_frame: (3040, 4064) uint16 array with Bayer data
        rgb_frame: (380, 508, 3) uint8 output array
        focus_mode: bool - True for focus mode, False for wide view
        draw_focus_bar: bool - True to draw focus bar overlay
        apply_color: bool - True to apply BLC + gain + CCM (ignored when focus_mode=True)
        apply_gain: bool - True to apply R/B gain before CCM (only used when apply_color=True)
        ccm: (3,3) float32 CCM matrix, pre-fetched outside JIT to avoid deadlock
    
    Returns:
        float: focus_score (0.0-1.0 range when focus_mode=True, 0.0 otherwise)
    """
    h_out, w_out = 380, 508
    focus_score = 0.0
    
    if focus_mode:
        # Focus mode: 2x2 binning of center 760x1016 crop
        # No color processing in focus mode
        row_start = 1140  # (3040 - 760) // 2
        col_start = 1524  # (4064 - 1016) // 2
        bin_size = 2
        shift_r = r_bit_depth - 8
        shift_g = g_bit_depth - 8 + 1 
        shift_b = b_bit_depth - 8
    else:
        # Wide mode: 8x8 binning of full frame
        row_start = 0
        col_start = 0
        bin_size = 8

        # bit depth - presentation depth + binning samples
        shift_r = r_bit_depth - 8 + 4
        shift_g = g_bit_depth - 8 + 4 + 1
        shift_b = b_bit_depth - 8 + 4

        if apply_color:
            # BLC per sample = 256 scaled to actual bit depth (256 at 12-bit, doubles per extra bit)
            # Scaled by bin count: 16 samples for R/B, 32 for G
            blc_sample_r = np.uint32(256) << np.uint32(r_bit_depth - 12)
            blc_sample_g = np.uint32(256) << np.uint32(g_bit_depth - 12)
            blc_sample_b = np.uint32(256) << np.uint32(b_bit_depth - 12)
            blc_r = blc_sample_r * np.uint32(16)
            blc_g = blc_sample_g * np.uint32(32)
            blc_b = blc_sample_b * np.uint32(16)
            # Saturation level = max possible sum - BLC
            sat_r = (np.uint32((1 << r_bit_depth) - 1) * np.uint32(16)) - blc_r
            sat_g = (np.uint32((1 << g_bit_depth) - 1) * np.uint32(32)) - blc_g
            sat_b = (np.uint32((1 << b_bit_depth) - 1) * np.uint32(16)) - blc_b

            # Pre-compute CCM-aware clip level at saturation (computed once before pixel loop)
            # Use sat_level at single-pixel scale as the saturated input to CCM
            sat_level = np.float32((1 << g_bit_depth) - 1) - np.float32(blc_sample_g)
            r_sat_ccm = ccm[0, 0] * sat_level + ccm[0, 1] * sat_level + ccm[0, 2] * sat_level
            g_sat_ccm = ccm[1, 0] * sat_level + ccm[1, 1] * sat_level + ccm[1, 2] * sat_level
            b_sat_ccm = ccm[2, 0] * sat_level + ccm[2, 1] * sat_level + ccm[2, 2] * sat_level
            # Use minimum so all channels clip to the same balanced ceiling
            clip_ccm = min(r_sat_ccm, min(g_sat_ccm, b_sat_ccm))

    # Main debayering loop
    for i in prange(h_out):
        for j in range(w_out):
            # Coordinates in original raw frame
            row = row_start + i * bin_size
            col = col_start + j * bin_size
                       
            # Use uint32 to avoid overflow
            r_sum = np.uint32(0)
            b_sum = np.uint32(0)
            g_sum = np.uint32(0)
            
            # Loop through the bin_size x bin_size block
            for dy in range(bin_size):
                for dx in range(bin_size):
                    pixel = np.uint32(raw_frame[row + dy, col + dx])
                    
                    # Determine pixel type from GBRG Bayer pattern
                    if dy % 2 == 0:  # Even rows
                        if dx % 2 == 0:
                            g_sum += pixel  # Green (top-left)
                        else:
                            b_sum += pixel  # Blue
                    else:  # Odd rows
                        if dx % 2 == 0:
                            r_sum += pixel  # Red
                        else:
                            g_sum += pixel  # Green (bottom-right)
            
            if apply_color and not focus_mode:
                # Apply BLC: subtract and clip to zero
                r_blc = r_sum - blc_r if r_sum > blc_r else np.uint32(0)
                g_blc = g_sum - blc_g if g_sum > blc_g else np.uint32(0)
                b_blc = b_sum - blc_b if b_sum > blc_b else np.uint32(0)

                # Clip to saturation level
                r_blc = min(r_blc, sat_r)
                g_blc = min(g_blc, sat_g)
                b_blc = min(b_blc, sat_b)

                # Convert to float at single-pixel scale
                r_f = np.float32(r_blc) / np.float32(16)
                g_f = np.float32(g_blc) / np.float32(32)
                b_f = np.float32(b_blc) / np.float32(16)

                if apply_gain:
                    # Apply gain with clip to sat_level = max - blc_per_sample
                    r_f = min(sat_level, r_f * gain_r)
                    g_f = min(sat_level, g_f)
                    b_f = min(sat_level, b_f * gain_b)

                # Apply CCM: [R', G', B'] = CCM x [R, G, B]
                r_ccm = ccm[0, 0] * r_f + ccm[0, 1] * g_f + ccm[0, 2] * b_f
                g_ccm = ccm[1, 0] * r_f + ccm[1, 1] * g_f + ccm[1, 2] * b_f
                b_ccm = ccm[2, 0] * r_f + ccm[2, 1] * g_f + ccm[2, 2] * b_f

                # Clip all channels to CCM-aware balanced ceiling
                r_ccm = max(np.float32(0), min(clip_ccm, r_ccm))
                g_ccm = max(np.float32(0), min(clip_ccm, g_ccm))
                b_ccm = max(np.float32(0), min(clip_ccm, b_ccm))

                # Shift from single-pixel bit depth to 8-bit
                shift_px_r = r_bit_depth - 8
                shift_px_g = g_bit_depth - 8
                shift_px_b = b_bit_depth - 8
                rgb_frame[i, j, 0] = np.uint8(r_ccm / np.float32(1 << shift_px_r))
                rgb_frame[i, j, 1] = np.uint8(g_ccm / np.float32(1 << shift_px_g))
                rgb_frame[i, j, 2] = np.uint8(b_ccm / np.float32(1 << shift_px_b))

            else:
                # No color processing: existing shift behaviour
                rgb_frame[i, j, 0] = np.uint8(r_sum >> shift_r)  # Red
                rgb_frame[i, j, 1] = np.uint8(g_sum >> shift_g)  # Green
                rgb_frame[i, j, 2] = np.uint8(b_sum >> shift_b)  # Blue
    
    # Calculate focus metric in focus mode
    if focus_mode:
        # Separate pass for focus calculation on raw green, red, and blue pixels
        crop_h = 760
        crop_w = 1016
        
        # Green channel focus calculation
        gradient_sum_g = 0.0
        intensity_sum_g = 0.0
        sample_count_g = 0
        
        # Red channel focus calculation
        gradient_sum_r = 0.0
        intensity_sum_r = 0.0
        sample_count_r = 0
        
        # Blue channel focus calculation
        gradient_sum_b = 0.0
        intensity_sum_b = 0.0
        sample_count_b = 0
        
        # Sample every 4th pixel for efficiency
        # In GBRG pattern:
        # - Greens are at (even,even) and (odd,odd)
        # - Reds are at (odd,even)
        # - Blues are at (even,odd)
        for row in range(row_start + 2, row_start + crop_h - 2, 4):
            for col in range(col_start + 2, col_start + crop_w - 2, 4):
                # Green at (even, even) positions in GBRG pattern
                center_g = np.float32(raw_frame[row, col])
                intensity_sum_g += center_g
                
                # Gradient with 2-pixel spacing (stays on green channel)
                left_g = np.float32(raw_frame[row, col - 2])
                right_g = np.float32(raw_frame[row, col + 2])
                top_g = np.float32(raw_frame[row - 2, col])
                bottom_g = np.float32(raw_frame[row + 2, col])
                
                dx_g = (right_g - left_g) / 4.0
                dy_g = (bottom_g - top_g) / 4.0
                
                gradient_sum_g += dx_g * dx_g + dy_g * dy_g
                sample_count_g += 1
                
                # Red at (odd, even) positions - offset by 1 row from green
                row_r = row + 1
                if row_r < row_start + crop_h - 2:
                    center_r = np.float32(raw_frame[row_r, col])
                    intensity_sum_r += center_r
                    
                    # Gradient with 2-pixel spacing (stays on red channel)
                    left_r = np.float32(raw_frame[row_r, col - 2])
                    right_r = np.float32(raw_frame[row_r, col + 2])
                    top_r = np.float32(raw_frame[row_r - 2, col])
                    bottom_r = np.float32(raw_frame[row_r + 2, col])
                    
                    dx_r = (right_r - left_r) / 4.0
                    dy_r = (bottom_r - top_r) / 4.0
                    
                    gradient_sum_r += dx_r * dx_r + dy_r * dy_r
                    sample_count_r += 1
                
                # Blue at (even, odd) positions - offset by 1 column from green
                col_b = col + 1
                if col_b < col_start + crop_w - 2:
                    center_b = np.float32(raw_frame[row, col_b])
                    intensity_sum_b += center_b
                    
                    # Gradient with 2-pixel spacing (stays on blue channel)
                    left_b = np.float32(raw_frame[row, col_b - 2])
                    right_b = np.float32(raw_frame[row, col_b + 2])
                    top_b = np.float32(raw_frame[row - 2, col_b])
                    bottom_b = np.float32(raw_frame[row + 2, col_b])
                    
                    dx_b = (right_b - left_b) / 4.0
                    dy_b = (bottom_b - top_b) / 4.0
                    
                    gradient_sum_b += dx_b * dx_b + dy_b * dy_b
                    sample_count_b += 1
        
        # Calculate green focus score
        focus_score_g = 0.0
        if sample_count_g > 0:
            mean_gradient_sq_g = gradient_sum_g / sample_count_g
            # Tenengrad: use square root of mean squared gradient
            focus_score_g = mean_gradient_sq_g ** 0.5 * 0.015
        
        # Calculate red focus score
        focus_score_r = 0.0
        if sample_count_r > 0:
            mean_gradient_sq_r = gradient_sum_r / sample_count_r
            focus_score_r = mean_gradient_sq_r ** 0.5 * 0.02
        
        # Calculate blue focus score
        focus_score_b = 0.0
        if sample_count_b > 0:
            mean_gradient_sq_b = gradient_sum_b / sample_count_b
            focus_score_b = mean_gradient_sq_b ** 0.5 * 0.03
        
        # Use green channel as primary focus score (highest resolution)
        focus_score = focus_score_g
        
        # Draw focus bars if requested
        if draw_focus_bar:
            bar_width = 10
            
            # Draw black background for all four bars
            black_bg = np.array([0, 0, 0], dtype=np.uint8)
            rgb_frame[:, :bar_width*4] = black_bg
            
            # Draw horizontal gray markers for reference (10 markers evenly spaced)
            marker_color = np.array([64, 64, 64], dtype=np.uint8)
            num_markers = 10
            for m in range(num_markers):
                marker_row = int((m + 1) * h_out / (num_markers + 1))
                rgb_frame[marker_row:marker_row+1, :bar_width*4] = marker_color
            
            # Calculate average focus score from normalized individual scores
            normalized_score_g = min(focus_score_g, 1.0)
            normalized_score_r = min(focus_score_r, 1.0)
            normalized_score_b = min(focus_score_b, 1.0)
            average_score = (normalized_score_g + normalized_score_r + normalized_score_b) / 3.0
            
            # Gray average bar (leftmost position)
            bar_height_avg = int(average_score * h_out)
            bar_height_avg = max(0, bar_height_avg)
            
            bar_color_avg = np.array([128, 128, 128], dtype=np.uint8)
            rgb_frame[h_out - bar_height_avg:h_out, :bar_width] = bar_color_avg
            
            # Green focus bar (second position)
            bar_height_g = int(normalized_score_g * h_out)
            bar_height_g = max(0, bar_height_g)
            
            bar_color_g = np.array([64, 128, 64], dtype=np.uint8)
            rgb_frame[h_out - bar_height_g:h_out, bar_width:bar_width*2] = bar_color_g
            
            # Red focus bar (third position)
            bar_height_r = int(normalized_score_r * h_out)
            bar_height_r = max(0, bar_height_r)
            
            bar_color_r = np.array([128, 64, 64], dtype=np.uint8)
            rgb_frame[h_out - bar_height_r:h_out, bar_width*2:bar_width*3] = bar_color_r
            
            # Blue focus bar (fourth position)
            bar_height_b = int(normalized_score_b * h_out)
            bar_height_b = max(0, bar_height_b)
            
            bar_color_b = np.array([64, 64, 128], dtype=np.uint8)
            rgb_frame[h_out - bar_height_b:h_out, bar_width*3:bar_width*4] = bar_color_b
    
    return focus_score


class DisplayDebayer:
    def __init__(self):
        log(LogTag.DEBAY, "Initializing...")
      
        # Get arrays using config keys
        self.flags = get_array("flags")
        self.rgb0_8 = get_array("shm_rgb0_8")
        self.rgb1_8 = get_array("shm_rgb1_8")
        self.raw0_16 = get_array("shm_raw0_16")
        self.raw1_16 = get_array("shm_raw1_16")
        self.raw_capture_16 = get_array("shm_raw_capture_16")
        self.sm_uint16var = get_array("shm_uint16_var")
        self.sm_floatvar  = get_array("shm_float_var")

        # Ensure gains are initialized before warm-up compilation
        # (in case auto_load_defaults ran before float_var was seeded)
        if self.sm_floatvar[FloatIndex.R_DIG_GAIN] == 0.0:
            self.sm_floatvar[FloatIndex.R_DIG_GAIN] = DEFAULT_R_DIG_GAIN
        if self.sm_floatvar[FloatIndex.B_DIG_GAIN] == 0.0:
            self.sm_floatvar[FloatIndex.B_DIG_GAIN] = DEFAULT_B_DIG_GAIN

        if (self.flags is None or self.raw0_16 is None or 
            self.raw1_16 is None):
            raise RuntimeError("Debayer 8x8: Failed to get shared arrays")
        
        log(LogTag.DEBAY, "Connected to shared memory")

        self.flags[FlagIndex.RGB_8_FOCUS] = False

        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R] = 12
        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G] = 12
        self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B] = 12

        self.r_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R]
        self.g_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G]
        self.b_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B]
        self.gain_r = float(self.sm_floatvar[FloatIndex.R_DIG_GAIN])
        self.gain_b = float(self.sm_floatvar[FloatIndex.B_DIG_GAIN])

        #Wait until previous module finished compiling
        while not self.flags[FlagIndex.MOD_READY_RAWCAPTURE]:
            time.sleep(0.5)

        log_compile_start(LogTag.DEBAY, "debayer function")
        compile_start = time.perf_counter()

        # Pre-fetch CCM matrix once in Python (passed into JIT to avoid deadlock)
        self.ccm = get_ccm_matrix(CCM_COLOR_TEMP)
        # Read gains from shared memory - same source used in process loop
        self.gain_r = float(self.sm_floatvar[FloatIndex.R_DIG_GAIN])
        self.gain_b = float(self.sm_floatvar[FloatIndex.B_DIG_GAIN])

        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, self.flags[FlagIndex.RGB_8_FOCUS], self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], False, False, self.ccm, self.gain_r, self.gain_b)
        debayer_with_focus(self.raw0_16, 12, 12, 12, self.rgb0_8, self.flags[FlagIndex.RGB_8_FOCUS], self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], False, False, self.ccm, self.gain_r, self.gain_b)
        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb1_8, self.flags[FlagIndex.RGB_8_FOCUS], self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], False, False, self.ccm, self.gain_r, self.gain_b)
        debayer_with_focus(self.raw1_16, 12, 12, 12, self.rgb1_8, self.flags[FlagIndex.RGB_8_FOCUS], self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], False, False, self.ccm, self.gain_r, self.gain_b)
        # Warm-up: color processing ON, no gain (buffer RGB capture)
        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, False, False, True, False, self.ccm, self.gain_r, self.gain_b)
        # Warm-up: color processing ON, with gain (buffer all-channel or live)
        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, False, False, True, True, self.ccm, self.gain_r, self.gain_b)
        debayer_with_focus(self.raw0_16, 12, 12, 12, self.rgb0_8, False, False, True, True, self.ccm, self.gain_r, self.gain_b)
        # Warm-up: focus_mode=True (Focus switch on) — different code path with 2x2 binning
        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, True, False, False, False, self.ccm, self.gain_r, self.gain_b)
        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, True, True, False, False, self.ccm, self.gain_r, self.gain_b)

        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.DEBAY, compile_time)
        log(LogTag.DEBAY, "Ready")

        #Raised finished compiling
        self.flags[FlagIndex.MOD_READY_DEBAYER] = True

        #Wait until last module finished compiling
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)

    def process(self):
        """Main processing loop"""
        log(LogTag.DEBAY, "Starting display loop...")
          
        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:
                self.r_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R]
                self.g_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G]
                self.b_bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B]
                self.gain_r = float(self.sm_floatvar[FloatIndex.R_DIG_GAIN])
                self.gain_b = float(self.sm_floatvar[FloatIndex.B_DIG_GAIN])

                # Process raw0_16 when ready
                if self.flags[FlagIndex.RQST_DBY_RGB0]:
                    self.flags[FlagIndex.RQST_DBY_RGB0] = False
                    focus_on = self.flags[FlagIndex.RGB_8_FOCUS]
                    color_on = self.flags[FlagIndex.RGB_8_COLOR_PROCESSING]
                    show_buffer = self.flags[FlagIndex.RGB_8_SHOW_BUFFER]
                    live_rgb = not show_buffer and self.flags[FlagIndex.MODE_CAPTURE_RGB]
                    apply_color = color_on and not focus_on and not live_rgb
                    if show_buffer:
                        buffer_capture_mode = self.raw_capture_16[4][-1]  # 0x0000=RGB, 0x0001=ALL
                        apply_gain = apply_color and (buffer_capture_mode != 0x0000)
                    else:
                        apply_gain = apply_color and not self.flags[FlagIndex.MODE_CAPTURE_RGB]
                    if show_buffer:
                        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb0_8, focus_on, self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], apply_color, apply_gain, self.ccm, self.gain_r, self.gain_b)
                    else:
                        debayer_with_focus(self.raw0_16, 12, 12, 12, self.rgb0_8, focus_on, self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], apply_color, apply_gain, self.ccm, self.gain_r, self.gain_b)

                    # Invert condensed display (post BLC/gain/CCM), affects waveform too
                    if self.flags[FlagIndex.RGB_8_INVERT]:
                        np.subtract(np.uint8(255), self.rgb0_8, out=self.rgb0_8)

                    # Request Waveform Processing:
                    self.flags[FlagIndex.RQST_WVF_1] = False
                    self.flags[FlagIndex.RQST_WVF_0] = True
                    self.flags[FlagIndex.READY_RGB1] = False
                    self.flags[FlagIndex.READY_RGB0] = True


                # Process blue1 when ready
                if self.flags[FlagIndex.RQST_DBY_RGB1]:
                    self.flags[FlagIndex.RQST_DBY_RGB1] = False
                    focus_on = self.flags[FlagIndex.RGB_8_FOCUS]
                    color_on = self.flags[FlagIndex.RGB_8_COLOR_PROCESSING]
                    show_buffer = self.flags[FlagIndex.RGB_8_SHOW_BUFFER]
                    live_rgb = not show_buffer and self.flags[FlagIndex.MODE_CAPTURE_RGB]
                    apply_color = color_on and not focus_on and not live_rgb
                    if show_buffer:
                        buffer_capture_mode = self.raw_capture_16[4][-1]  # 0x0000=RGB, 0x0001=ALL
                        apply_gain = apply_color and (buffer_capture_mode != 0x0000)
                    else:
                        apply_gain = apply_color and not self.flags[FlagIndex.MODE_CAPTURE_RGB]
                    if show_buffer:
                        debayer_with_focus(self.raw_capture_16, self.r_bit_depth, self.g_bit_depth, self.b_bit_depth, self.rgb1_8, focus_on, self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], apply_color, apply_gain, self.ccm, self.gain_r, self.gain_b)
                    else:
                        debayer_with_focus(self.raw1_16, 12, 12, 12, self.rgb1_8, focus_on, self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR], apply_color, apply_gain, self.ccm, self.gain_r, self.gain_b)

                    # Invert condensed display (post BLC/gain/CCM), affects waveform too
                    if self.flags[FlagIndex.RGB_8_INVERT]:
                        np.subtract(np.uint8(255), self.rgb1_8, out=self.rgb1_8)

                    # Request Waveform Processing:
                    self.flags[FlagIndex.RQST_WVF_0] = False
                    self.flags[FlagIndex.RQST_WVF_1] = True
                    self.flags[FlagIndex.READY_RGB0] = False
                    self.flags[FlagIndex.READY_RGB1] = True

                    
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            log(LogTag.DEBAY, "Interrupted by user")
        except Exception as e:
            log(LogTag.DEBAY, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()
        finally:
            log(LogTag.DEBAY, "Stopping...")


def main():
    """Main function for Debayer 8x8 module"""
    detector = None
    try:
        debayer = DisplayDebayer()
        debayer.process()
    except Exception as e:
        log(LogTag.DEBAY, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.DEBAY, "Shutting down...")
        cleanup_manager()
        log(LogTag.DEBAY, "Cleanup complete")


if __name__ == "__main__":
    main()