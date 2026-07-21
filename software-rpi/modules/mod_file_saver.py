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
File Saver — TIFF output and embedded-metadata handling.

Receives save requests and writes frames to disk as 16-bit TIFFs, either raw
sensor data or debayered/cropped RGB. Debayers via the shared BLC+CCM routine,
applies optional marker-referenced cropping, and re-embeds frame metadata
(V/H position, marker, crop, mode, gains) into the output so it survives
independently of GUI state. Saves run on a background worker thread fed by a
bounded queue.

Shared memory:
    Reads:  Flags — SHUTDOWN_REQUESTED, SAVE_REQUEST, SAVE_RAW,
            SAVE_APPLY_CROP, MOD_READY_WAVEFORM, MOD_READY_ALL;
            Uint16 — CAPTURE_DEPTH_R/G/B; Float — R_DIG_GAIN, B_DIG_GAIN;
            String — FILE_PATH, METADATA_1, METADATA_2, USER_COMMENT;
            raw_capture (shm_raw_capture_16, incl. embedded metadata column)
    Writes: Flags — MOD_READY_FILESAVER, SAVE_COMPLETE,
            SAVE_COMPLETE_CONTROLLER; Float — R_DIG_GAIN, B_DIG_GAIN (seed if
            unset); String — USER_COMMENT (frame-number increment)

Interacts with:
    mod_controller (SAVE_REQUEST/SAVE_COMPLETE_CONTROLLER), mod_gui
    (SAVE_RAW, SAVE_APPLY_CROP, FILE_PATH/METADATA/USER_COMMENT strings),
    mod_waveform (MOD_READY_WAVEFORM gate), mod_debayer_with_blc_ccm
    (debayer_stride2_with_blc_ccm), shm_manager (arrays)
"""
import time
import threading
from queue import Queue, Empty
from pathlib import Path

import numpy as np
import tifffile
from numba import njit

from modules.memory.shm_manager import get_array, cleanup_manager, read_string, write_string
from modules.memory.shm_indexing import FlagIndex, Uint16Index, StringIndex, FloatIndex
from modules.mod_debayer_with_blc_ccm import debayer_stride2_with_blc_ccm
from tpx_config import (DEFAULT_R_DIG_GAIN, DEFAULT_B_DIG_GAIN,
                        CCM_COLOR_TEMP, BLACK_LEVEL_R, BLACK_LEVEL_G, BLACK_LEVEL_B)
from tpx_logger import log, LogTag, log_compile_start, log_compile_done


@njit(cache=True)
def apply_crop(rgb_input, rgb_output, top, bottom, left, right):
    """Copy the crop region defined by absolute top/bottom/left/right pixel
    positions from the input RGB image into rgb_output, clamping the bounds to
    the input dimensions."""
    h_in, w_in, _ = rgb_input.shape

    top = max(0, min(top, h_in))
    bottom = max(0, min(bottom, h_in))
    left = max(0, min(left, w_in))
    right = max(0, min(right, w_in))

    for i in range(bottom - top):
        for j in range(right - left):
            for c in range(3):
                rgb_output[i, j, c] = rgb_input[top + i, left + j, c]

class FileSaverProcessor:
    def __init__(self):
        log(LogTag.FSAVE, "Initializing...")

        self.flags = get_array("flags")
        self.raw_capture = get_array("shm_raw_capture_16")

        self.sm_uint16var = get_array("shm_uint16_var")
        self.sm_floatvar  = get_array("shm_float_var")

        if self.flags is None or self.raw_capture is None or self.sm_uint16var is None or self.sm_floatvar is None:
            raise RuntimeError("File Saver: Failed to get shared arrays")

        log(LogTag.FSAVE, "Connected to shared memory")

        # Ensure gains are initialized before warm-up
        if self.sm_floatvar[FloatIndex.R_DIG_GAIN] == 0.0:
            self.sm_floatvar[FloatIndex.R_DIG_GAIN] = DEFAULT_R_DIG_GAIN
        if self.sm_floatvar[FloatIndex.B_DIG_GAIN] == 0.0:
            self.sm_floatvar[FloatIndex.B_DIG_GAIN] = DEFAULT_B_DIG_GAIN

        # Default crop values
        self.crop_height=1200
        self.crop_top=160
        self.crop_left=100
        self.crop_right=100
        self.crop_bottom = 100

        self.rgb_debayered = np.zeros((1520, 2032, 3), dtype=np.uint16)

        self.bit_depth_r = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R]
        self.bit_depth_g = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G]
        self.bit_depth_b = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B]

        # Save queue for threaded saving
        self.save_queue = Queue(maxsize=2)
        self.worker_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.worker_thread.start()

        # Wait for the previous module to finish compiling
        while not self.flags[FlagIndex.MOD_READY_WAVEFORM]:
            time.sleep(0.5)

        log_compile_start(LogTag.FSAVE, "JIT functions")
        compile_start = time.perf_counter()

        _gain_r = np.float32(self.sm_floatvar[FloatIndex.R_DIG_GAIN])
        _gain_b = np.float32(self.sm_floatvar[FloatIndex.B_DIG_GAIN])

        # Warm-up: exercise both apply_gain paths so the JIT signatures are
        # compiled before the first real save.
        debayer_stride2_with_blc_ccm(
                self.raw_capture,
                bit_depth_r=self.bit_depth_r,
                bit_depth_g=self.bit_depth_g,
                bit_depth_b=self.bit_depth_b,
                rgb_output=self.rgb_debayered,
                black_r=BLACK_LEVEL_R,
                black_g=BLACK_LEVEL_G,
                black_b=BLACK_LEVEL_B,
                color_temp=CCM_COLOR_TEMP,
                apply_ccm=True,
                apply_gain=False,
                gain_r=_gain_r,
                gain_b=_gain_b,
            )
        debayer_stride2_with_blc_ccm(
                self.raw_capture,
                bit_depth_r=self.bit_depth_r,
                bit_depth_g=self.bit_depth_g,
                bit_depth_b=self.bit_depth_b,
                rgb_output=self.rgb_debayered,
                black_r=BLACK_LEVEL_R,
                black_g=BLACK_LEVEL_G,
                black_b=BLACK_LEVEL_B,
                color_temp=CCM_COLOR_TEMP,
                apply_ccm=True,
                apply_gain=True,
                gain_r=_gain_r,
                gain_b=_gain_b,
            )

        # Warm-up crop (actual crop size changes at save time from GUI state)
        self.rgb_cropped = np.zeros((1300, 1520, 3), dtype=np.uint16)
        self.crop_left, self.crop_right, self.crop_top, self.crop_bottom = 0,0,0,0
        apply_crop(self.rgb_debayered, self.rgb_cropped, self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)

        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.FSAVE, compile_time)

        self.flags[FlagIndex.MOD_READY_FILESAVER] = True

        # Wait for all modules to finish compiling
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)

        log(LogTag.FSAVE, "Ready")

    def _save_worker(self):
        """Background worker: pull queued (filepath, data) jobs and write each
        to disk as a TIFF, then advance the frame number and signal
        completion. Runs until shutdown is requested."""
        while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:
            try:
                save_job = self.save_queue.get(timeout=1.0)
                filepath, data = save_job

                # Blocking write, but off the main loop on this worker thread
                start_time = time.perf_counter()
                tifffile.imwrite(filepath, data, compression=None)
                save_time = (time.perf_counter() - start_time) * 1000

                log(LogTag.FSAVE, f"Saved {filepath} ({save_time:.1f}ms)")

                self._increment_frame_number()
                self.flags[FlagIndex.SAVE_COMPLETE] = True

            except Empty:
                continue
            except Exception as e:
                log(LogTag.FSAVE, f"Error in worker thread — {e}", level="ERR")

    def _increment_frame_number(self):
        """Advance the zero-padded 6-digit frame number stored in the
        USER_COMMENT string, used to name the next saved file."""
        try:
            frame_str = read_string(StringIndex.USER_COMMENT)
            if frame_str and frame_str.isdigit():
                frame_num = int(frame_str)
                frame_num += 1
                write_string(StringIndex.USER_COMMENT, f"{frame_num:06d}")
                log(LogTag.FSAVE, f"Frame counter → {frame_num:06d}")
        except Exception as e:
            log(LogTag.FSAVE, f"Error incrementing frame counter — {e}", level="ERR")

    def _build_filepath(self):
        """Assemble the output path from the GUI-provided path, prefix, scene,
        and frame strings, falling back to defaults for any that are empty, and
        ensure the target directory exists."""
        filepath = read_string(StringIndex.FILE_PATH)
        prefix = read_string(StringIndex.METADATA_1)  # reused for prefix
        scene = read_string(StringIndex.METADATA_2)   # reused for scene
        frame = read_string(StringIndex.USER_COMMENT) # reused for frame

        if not filepath:
            filepath = str(Path.home() / "tingopix_reels")
        if not prefix:
            prefix = "tpx"
        if not scene:
            scene = "001"
        if not frame:
            frame = "000001"

        filename = f"{prefix}_{scene}_{frame}.tiff"
        full_path = Path(filepath) / filename

        full_path.parent.mkdir(parents=True, exist_ok=True)

        return str(full_path)

    def save_raw(self):
        """Save the raw sensor capture as a 16-bit TIFF (3040x4064), shifting
        the sensor bits up to the MSBs and re-embedding the frame metadata into
        the last column so it survives in the saved file."""
        log(LogTag.FSAVE, "Starting raw save...")

        # Use the R channel bit depth as the reference for the raw shift
        bit_depth = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R]
        shift_amount = 16 - bit_depth

        # Extract embedded position/mode from the capture array
        film_position = self.raw_capture [0][-1]
        film_edge = self.raw_capture [1][-1]
        marker_position = self.raw_capture [2][-1]
        crop_height = self.raw_capture [3][-1]
        capture_mode = self.raw_capture [4][-1]

        raw_copy = self.raw_capture.copy()
        if shift_amount > 0:
            raw_copy = raw_copy << shift_amount

        # Array is already (3040, 4064) = height x width; no transpose needed

        filepath = self._build_filepath()

        # Re-embed position/mode/gains into the saved copy
        raw_copy[0][-1] = film_position
        raw_copy[1][-1] = film_edge
        raw_copy[2][-1] = marker_position
        raw_copy[3][-1] = crop_height
        raw_copy[4][-1] = capture_mode
        raw_copy[5][-1] = self.raw_capture[5][-1]  # gain_r * 1000
        raw_copy[6][-1] = self.raw_capture[6][-1]  # gain_b * 1000

        log(LogTag.FSAVE, f"Raw save — V:{film_position} H:{film_edge} marker:{marker_position} | {bit_depth}→16bit (3040×4064)")

        try:
            self.save_queue.put_nowait((filepath, raw_copy))
        except:
            log(LogTag.FSAVE, "Save queue full — skipping frame", level="WARN")

    def calculate_crop_params(self, film_position, marker_position, crop_height, film_edge, frame_height):
        """Compute the crop rectangle for the binned frame using marker-
        referenced vertical positioning and a fixed-width horizontal window
        anchored at the film edge, clamping to the frame bounds. Inputs are in
        full-resolution coordinates; output is in binned coordinates."""
        # Convert from full-resolution to binned coordinates for calculation
        film_position = film_position // 2
        film_edge_absolute = film_edge // 2
        marker_position = marker_position // 2
        crop_height = crop_height // 2

        # Now convert film_edge_absolute back to AOI-relative
        crop_width = 1904
        aoi_left = 1904
        film_edge = film_edge_absolute - aoi_left

        # Convert film_edge from AOI coordinates to full frame coordinates
        film_edge_absolute = aoi_left + film_edge

        # Vertical: center crop, then shift to keep marker-referenced position
        frame_vertical_shift = film_position - marker_position
        top_crop = (frame_height - crop_height) // 2 + frame_vertical_shift
        bottom_crop = top_crop + crop_height

        # Clamp vertical bounds
        if top_crop < 0:
            log(LogTag.FSAVE, f"Crop: top_crop clamped from {top_crop} to 0", level="WARN")
            top_crop = 0
            bottom_crop = crop_height
        elif bottom_crop > frame_height:
            log(LogTag.FSAVE, f"Crop: bottom_crop clamped from {bottom_crop} to {frame_height}", level="WARN")
            bottom_crop = frame_height
            top_crop = frame_height - crop_height

        # Horizontal: fixed width with right edge at film_edge
        right_crop = film_edge_absolute
        left_crop = film_edge_absolute - crop_width

        # Clamp horizontal bounds
        if left_crop < 0:
            log(LogTag.FSAVE, f"Crop: left_crop clamped from {left_crop} to 0", level="WARN")
            left_crop = 0
            right_crop = crop_width
        elif right_crop > 2032:
            log(LogTag.FSAVE, f"Crop: right_crop clamped from {right_crop} to 2032", level="WARN")
            right_crop = 2032
            left_crop = 2032 - crop_width

        return left_crop, right_crop, top_crop, bottom_crop

    def save_debayered(self):
        """Debayer the capture (BLC+CCM), apply optional marker-referenced
        cropping, re-embed the frame metadata into the blue channel so it
        survives the crop, and queue the RGB frame for saving as a TIFF."""
        log(LogTag.FSAVE, "Starting debayered save...")

        self.bit_depth_r = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_R]
        self.bit_depth_g = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_G]
        self.bit_depth_b = self.sm_uint16var[Uint16Index.CAPTURE_DEPTH_B]

        # Extract embedded position and gains from the capture array
        film_position  = self.raw_capture[0][-1]
        film_edge      = self.raw_capture[1][-1]
        marker_position = self.raw_capture[2][-1]
        crop_height    = self.raw_capture[3][-1]
        capture_mode   = self.raw_capture[4][-1]
        gain_r_embed   = self.raw_capture[5][-1]  # gain_r * 1000
        gain_b_embed   = self.raw_capture[6][-1]  # gain_b * 1000

        _gain_r = np.float32(self.sm_floatvar[FloatIndex.R_DIG_GAIN])
        _gain_b = np.float32(self.sm_floatvar[FloatIndex.B_DIG_GAIN])

        if (capture_mode == 0x0000) : #RGB CAPTURE
            debayer_stride2_with_blc_ccm(
                    self.raw_capture,
                    bit_depth_r=self.bit_depth_r,
                    bit_depth_g=self.bit_depth_g,
                    bit_depth_b=self.bit_depth_b,
                    rgb_output=self.rgb_debayered,
                    black_r=BLACK_LEVEL_R,
                    black_g=BLACK_LEVEL_G,
                    black_b=BLACK_LEVEL_B,
                    color_temp=CCM_COLOR_TEMP,
                    apply_ccm=True,
                    apply_gain=False,
                    gain_r=_gain_r,
                    gain_b=_gain_b,
                )
        else: #ALL CHANNEL CAPTURE
            # NOTE: apply_ccm=True applies gain correction and CCM to all-channel captures.
            # gain_r/gain_b are read from shared memory (set via GUI gain panel)
            debayer_stride2_with_blc_ccm(
                    self.raw_capture,
                    bit_depth_r=self.bit_depth_r,
                    bit_depth_g=self.bit_depth_g,
                    bit_depth_b=self.bit_depth_b,
                    rgb_output=self.rgb_debayered,
                    black_r=BLACK_LEVEL_R,
                    black_g=BLACK_LEVEL_G,
                    black_b=BLACK_LEVEL_B,
                    color_temp=CCM_COLOR_TEMP,
                    apply_ccm=True,
                    apply_gain=True,
                    gain_r=_gain_r,
                    gain_b=_gain_b,
                )

        # Embed position into the debayered RGB array (blue channel)
        self.rgb_debayered[0][-1][2] =  film_position
        self.rgb_debayered[1][-1][2] = film_edge
        self.rgb_debayered[2][-1][2] = marker_position
        self.rgb_debayered[3][-1][2] = crop_height

        log(LogTag.FSAVE, f"Debayered save — V:{film_position} H:{film_edge} marker:{marker_position} mode:{'RGB' if capture_mode == 0x0000 else 'ALL'}")

        # If no sprocket detected (sentinel 0xFFFF), skip crop entirely
        if film_position == 0xFFFF or film_edge == 0xFFFF:
            log(LogTag.FSAVE, "No sprocket detection — crop skipped, saving full frame", level="WARN")
            self.crop_left, self.crop_right, self.crop_top, self.crop_bottom = 0, 0, 0, 0
        else:
            # Cast through signed int16 for the crop math
            self.crop_left, self.crop_right, self.crop_top, self.crop_bottom = self.calculate_crop_params(
                int(np.int16(film_position)),
                int(np.int16(marker_position)),
                int(np.int16(crop_height)),
                int(np.int16(film_edge)),
                1520
            )

        # Apply cropping only if enabled AND crop values are non-zero
        apply_crop_enabled = self.flags[FlagIndex.SAVE_APPLY_CROP]
        if apply_crop_enabled and (self.crop_top > 0 or self.crop_bottom > 0 or self.crop_left > 0 or self.crop_right > 0):
            h_out = self.crop_bottom - self.crop_top
            w_out = self.crop_right - self.crop_left

            self.rgb_cropped = np.zeros((h_out, w_out, 3), dtype=np.uint16)
            apply_crop(self.rgb_debayered, self.rgb_cropped, self.crop_top, self.crop_bottom, self.crop_left, self.crop_right)
            save_data = self.rgb_cropped
            log(LogTag.FSAVE, f"Crop applied — output ({h_out}×{w_out})")
        else:
            save_data = self.rgb_debayered.copy()
            if not apply_crop_enabled:
                log(LogTag.FSAVE, "Crop disabled — saving full frame (1520×2032)")
            else:
                log(LogTag.FSAVE, "No crop params set — saving full frame (1520×2032)")

        # Embed metadata into blue channel last column (after crop so it always survives)
        save_data[0][-1][2] = film_position
        save_data[1][-1][2] = film_edge
        save_data[2][-1][2] = marker_position
        save_data[3][-1][2] = crop_height
        save_data[4][-1][2] = capture_mode
        save_data[5][-1][2] = gain_r_embed   # gain_r * 1000
        save_data[6][-1][2] = gain_b_embed   # gain_b * 1000

        filepath = self._build_filepath()

        try:
            self.save_queue.put_nowait((filepath, save_data))
            log(LogTag.FSAVE, f"Debayered save queued — R:{self.bit_depth_r} G:{self.bit_depth_g} B:{self.bit_depth_b} bits")
        except:
            log(LogTag.FSAVE, "Save queue full — skipping frame", level="WARN")

    def process(self):
        """Main service loop: wait for a save request, dispatch to the raw or
        debayered save path, and signal completion until shutdown is
        requested."""
        log(LogTag.FSAVE, "Starting main loop...")

        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                if self.flags[FlagIndex.SAVE_REQUEST]:
                    self.flags[FlagIndex.SAVE_REQUEST] = False
                    self.flags[FlagIndex.SAVE_COMPLETE] = False
                    self.flags[FlagIndex.SAVE_COMPLETE_CONTROLLER] = False

                    if self.flags[FlagIndex.SAVE_RAW]:
                        self.save_raw()
                    else:
                        self.save_debayered()
                    self.flags[FlagIndex.SAVE_COMPLETE] = True
                    self.flags[FlagIndex.SAVE_COMPLETE_CONTROLLER] = True

                time.sleep(0.01)

        except KeyboardInterrupt:
            log(LogTag.FSAVE, "Interrupted by user")
        except Exception as e:
            log(LogTag.FSAVE, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()
        finally:
            log(LogTag.FSAVE, "Stopping...")


def main():
    """Process entry point: build the File Saver and run its loop, releasing
    shared memory on exit."""
    saver = None
    try:
        saver = FileSaverProcessor()
        saver.process()
    except Exception as e:
        log(LogTag.FSAVE, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.FSAVE, "Shutting down...")
        cleanup_manager()
        log(LogTag.FSAVE, "Cleanup complete")


if __name__ == "__main__":
    main()