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
Sprocket hole detector

Locates sprocket holes and the film edge in the blue-channel AOI to drive
frame registration. From the AOI slice, builds a vertical projection profile,
finds one or two holes by brightness with edge and horizontal-edge detection,
writes hole geometry into the sprocket-position arrays, and derives the film's
vertical/horizontal location and step distance to the alignment marker.

Shared memory:
    Reads:  MOD_READY_SPROCKET, SHUTDOWN_REQUESTED, SPROCKET_HALT,
            RQST_SPK_B0, RQST_SPK_B1, AOI_SPROCKET_LOCATION, SPROCKET_MARKER,
            PIX_TO_STEPS, blue0_8, blue1_8
    Writes: MOD_READY_DETECTOR, SPROCKET_HALT (init), RQST_SPK_B0/B1 (clears),
            READY_AOI0, READY_AOI1, TO_MARKER_FWD, AOI_SPROCKET_LOCATION (init),
            FILM_LOCATION_VERTICAL, FILM_LOCATION_HORIZONTAL, STEPS_TO_MARKER,
            sprocket_position_blue0, sprocket_position_blue1

Interacts with:
    mod_sprocket    (RQST_SPK_B0/B1, blue0_8/blue1_8)
    mod_controller  (FILM_LOCATION_VERTICAL, STEPS_TO_MARKER, TO_MARKER_FWD)
    mod_raw_capture (FILM_LOCATION_HORIZONTAL, via the metadata-embed path)
    gui_aoi_panel   (READY_AOI0/1)
    mod_gui         (SPROCKET_HALT, AOI_SPROCKET_LOCATION)
    shm_manager     (arrays)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, SprocketIndex, Uint16Index
from tpx_logger import log, LogTag, log_compile_start, log_compile_done

import time
import numpy as np
from numba import njit


@njit(cache=True)
def detect_film_edge(blue_aoi, y_top, y_bottom, edge_start_x, edge_end_x):
    """
    Detect the film edge in the horizontal direction.
    Uses same approach as sprocket detection - threshold from brightest point.
    
    Args:
        blue_aoi: Input array
        y_top: Top boundary of hole
        y_bottom: Bottom boundary of hole
        edge_start_x: Start of edge detection region (256)
        edge_end_x: End of edge detection region (384)
    
    Returns:
        X-position of film edge, or 0xFFFF if not found
    """
    # Validate input boundaries
    if y_top >= y_bottom or y_top < 0 or y_bottom > blue_aoi.shape[0]:
        return 0xFFFF
    
    # Use the center vertical position of the hole
    y_sample = (y_top + y_bottom) // 2
    
    # Create horizontal profile in edge region
    profile_length = edge_end_x - edge_start_x
    h_profile = np.zeros(profile_length, dtype=np.int32)
    
    # Sample multiple rows around center for robustness
    sample_range = min(20, (y_bottom - y_top) // 4)
    y_start_sample = max(y_top, y_sample - sample_range)
    y_end_sample = min(y_bottom, y_sample + sample_range)
    
    # Ensure valid range
    if y_start_sample >= y_end_sample:
        return 0xFFFF
    
    for y in range(y_start_sample, y_end_sample):
        for x in range(profile_length):
            h_profile[x] += blue_aoi[y, edge_start_x + x]
    
    # Find brightest point (no-film area / light)
    max_brightness_pos = 0
    max_value = h_profile[0]
    for x in range(profile_length):
        if h_profile[x] > max_value:
            max_value = h_profile[x]
            max_brightness_pos = x
    
    # Check minimum brightness threshold
    num_rows = y_end_sample - y_start_sample
    min_threshold = num_rows * 10
    if max_value < min_threshold:
        return 0xFFFF
    
    # Threshold: 15% below peak (same as sprocket detection)
    threshold = (max_value * 85) // 100
    
    # Scan from bright area (no film) toward left to find edge
    # Edge is where brightness drops below threshold
    edge_x = 0xFFFF
    for x in range(max_brightness_pos, -1, -1):
        if h_profile[x] < threshold:
            edge_x = edge_start_x + x
            break
    
    return edge_x


@njit(cache=True)
def detect_holes(blue_aoi, sprocket_position, frame_idx, hole_x_start=5, hole_x_end=105):
    """
    Detect sprocket holes and film edges in blue channel AOI.
    """
    height = 1520
    width_edge_start = 256
    width_total = 384
    level_threshold = 20

    
    POS_HOLE0 = 0
    POS_HOLE1 = 1
    POS_TOP = 0
    POS_CENTER = 1
    POS_BOTTOM = 2
    POS_EDGE = 3

    # Step 1: Create vertical projection profile from hole region only
    profile = np.zeros(height, dtype=np.int32)
    for y in range(height):
        for x in range(hole_x_start, hole_x_end, 4):
            profile[y] += blue_aoi[y, x]
    
    # Step 2: Find first hole (brightest region)
    hole1_center = np.argmax(profile)
    hole1_brightness = profile[hole1_center]
    
    # Calculate minimum threshold based on columns sampled
    num_columns = (hole_x_end - hole_x_start) // 4
    min_threshold = num_columns * 40 # Higher number = higher threshold ### Threshold for Level

    if hole1_brightness < min_threshold:
        # No valid detection
        for hole_idx in range(2):
            for pos_idx in range(4):
                sprocket_position[frame_idx, hole_idx, pos_idx] = 0xFFFF

        film_position = 0xFFFF
        edge_position = 0xFFFF
        return film_position, edge_position
    
    # Step 3: Find hole1 edges
    threshold1 = (hole1_brightness * (100-level_threshold)) // 100
    search_range = 400
    
    # Find TOP edge - scan upward from center
    hole1_top = 0
    for y in range(hole1_center - 1, max(0, hole1_center - search_range) - 1, -1):
        if profile[y] < threshold1:
            hole1_top = y + 1
            break
    
    # Find BOTTOM edge - scan downward from center  
    hole1_bottom = height - 1
    for y in range(hole1_center + 1, min(height, hole1_center + search_range)):
        if profile[y] < threshold1:
            hole1_bottom = y - 1
            break

    hole1_edge_x = detect_film_edge(blue_aoi, hole1_top, hole1_bottom, 
                                    width_edge_start, width_total)
    
    # Store hole1
    sprocket_position[frame_idx, POS_HOLE0, POS_TOP] = hole1_top
    sprocket_position[frame_idx, POS_HOLE0, POS_BOTTOM] = hole1_bottom
    sprocket_position[frame_idx, POS_HOLE0, POS_CENTER] = (hole1_top+hole1_bottom) // 2 
    
    sprocket_position[frame_idx, POS_HOLE0, POS_EDGE] = hole1_edge_x
    
    # Step 4: Mask hole1 based on DETECTED EDGES (not center)
    masked_profile = profile.copy()
    
    safety_margin = 50
    # Ensure mask boundaries are within array bounds
    mask_start = max(0, hole1_top - safety_margin)
    mask_end = min(height, hole1_bottom + safety_margin)
    
    for i in range(mask_start, mask_end):
        masked_profile[i] = 0
    
    # Step 5: Find second hole
    hole2_center = np.argmax(masked_profile)
    hole2_brightness = masked_profile[hole2_center]
    
    # Validate hole2
    min_relative_brightness = (hole1_brightness * 95) // 100 # Was 80 prior
    min_separation = 200
    
    if (hole2_brightness > min_relative_brightness and 
        hole2_brightness > min_threshold and
        abs(hole2_center - hole1_center) > min_separation):

        ### TWO HOLES CASE
        
        # Find hole2 edges
        threshold2 = (hole2_brightness * (100-level_threshold)) // 100
        
        hole2_top = 0
        for y in range(hole2_center - 1, max(0, hole2_center - search_range) - 1, -1):
            if profile[y] < threshold2:
                hole2_top = y + 1
                break
        
        hole2_bottom = height - 1
        for y in range(hole2_center + 1, min(height, hole2_center + search_range)):
            if profile[y] < threshold2:
                hole2_bottom = y - 1
                break


        hole2_edge_x = detect_film_edge(blue_aoi, hole2_top, hole2_bottom,
                                        width_edge_start, width_total)

        #IF TWO HOLES ARE DETECTED, KEEP THE ORDER

        if(hole1_top < hole2_top):

            sprocket_position[frame_idx, POS_HOLE1, POS_TOP] = hole2_top 
            sprocket_position[frame_idx, POS_HOLE1, POS_BOTTOM] = hole2_bottom
            sprocket_position[frame_idx, POS_HOLE1, POS_CENTER] = (hole2_top + hole2_bottom) // 2
            sprocket_position[frame_idx, POS_HOLE1, POS_EDGE] = hole2_edge_x

            sprocket_position[frame_idx, POS_HOLE0, POS_TOP] = hole1_top
            sprocket_position[frame_idx, POS_HOLE0, POS_BOTTOM] = hole1_bottom
            sprocket_position[frame_idx, POS_HOLE0, POS_CENTER] = (hole1_top + hole1_bottom) // 2
            sprocket_position[frame_idx, POS_HOLE0, POS_EDGE] = hole1_edge_x

            film_position = hole2_top
            edge_position = hole2_edge_x

        else:
            sprocket_position[frame_idx, POS_HOLE1, POS_TOP] = hole1_top 
            sprocket_position[frame_idx, POS_HOLE1, POS_BOTTOM] = hole1_bottom
            sprocket_position[frame_idx, POS_HOLE1, POS_CENTER] = (hole1_top + hole1_bottom) // 2
            sprocket_position[frame_idx, POS_HOLE1, POS_EDGE] = hole1_edge_x

            sprocket_position[frame_idx, POS_HOLE0, POS_TOP] = hole2_top
            sprocket_position[frame_idx, POS_HOLE0, POS_BOTTOM] = hole2_bottom
            sprocket_position[frame_idx, POS_HOLE0, POS_CENTER] = (hole2_top + hole2_bottom) // 2
            sprocket_position[frame_idx, POS_HOLE0, POS_EDGE] = hole2_edge_x

            film_position = hole1_top
            edge_position = hole1_edge_x

    else:

        ### ONE HOLES CASE
        for pos_idx in range(4):
            sprocket_position[frame_idx, POS_HOLE1, pos_idx] = 0xFFFF
        film_position = hole1_top
        edge_position = hole1_edge_x



    
    return film_position, edge_position

class SprocketDetector:
    def __init__(self):
        log(LogTag.DET, "Initializing...")

        # Get arrays using config keys
        self.flags = get_array("flags")
        self.blue0_8 = get_array("shm_blue0_8")
        self.blue1_8 = get_array("shm_blue1_8")
        self.sprocket_position_blue0 = get_array("shm_sprocket_position_blue0")
        self.sprocket_position_blue1 = get_array("shm_sprocket_position_blue1")
        self.sm_uint16var = get_array("shm_uint16_var")

        self.flags[FlagIndex.SPROCKET_HALT]=False

        if (self.flags is None or self.blue0_8 is None or 
            self.blue1_8 is None or self.sprocket_position_blue0 is None or self.sprocket_position_blue1 is None):
            raise RuntimeError("Sprocket Detector: Failed to get shared arrays")
        
        log(LogTag.DET, "Connected to shared memory")

        # Initialize AOI_SPROCKET_LOCATION default in shared memory if not yet set
        if self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] == 0:
            self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] = 5

        # Read aoi for film from shared memory — same path as runtime, ensures
        # uint16 type at warmup so Numba doesn't recompile on first runtime call
        self.hole_x_start = self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION]
        self.hole_x_end = self.hole_x_start + 128

        #Wait until previous module finished compiling
        while not self.flags[FlagIndex.MOD_READY_SPROCKET]:
            time.sleep(0.5)

        log_compile_start(LogTag.DET, "hole detection function")
        compile_start = time.perf_counter()

        detect_holes(self.blue0_8, self.sprocket_position_blue0, SprocketIndex.POS_FRAME0,
                        self.hole_x_start, self.hole_x_end)
        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.DET, compile_time)
        log(LogTag.DET, "Ready")

        #Raised finished compiling
        self.flags[FlagIndex.MOD_READY_DETECTOR] = True


    def process(self):
        """Main processing loop"""
        log(LogTag.DET, "Starting detection loop...")
          
        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:
                if not self.flags[FlagIndex.SPROCKET_HALT]:

                    #aoi for film
                    self.hole_x_start = self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION]
                    self.hole_x_end = self.hole_x_start + 128


                    # Process blue0 when ready
                    if self.flags[FlagIndex.RQST_SPK_B0]:
                        self.flags[FlagIndex.RQST_SPK_B0] = False

                        # Detection writes directly to shared memory
                        film_position, edge_position = detect_holes(self.blue0_8, self.sprocket_position_blue0, 
                                                    SprocketIndex.POS_FRAME0,
                                                    self.hole_x_start, self.hole_x_end)

                        self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL] = film_position
                        self.sm_uint16var[Uint16Index.FILM_LOCATION_HORIZONTAL] = edge_position - 256 if edge_position < 0xFFFF else 0xFFFF


                        if film_position < 0xFFFF:
                            self.sm_uint16var[Uint16Index.STEPS_TO_MARKER] = int(abs(float(film_position) - float(self.sm_uint16var[Uint16Index.SPROCKET_MARKER])) * float(self.sm_uint16var[Uint16Index.PIX_TO_STEPS]) / 1000.0)
                            self.flags[FlagIndex.TO_MARKER_FWD] = film_position > self.sm_uint16var[Uint16Index.SPROCKET_MARKER]

                        self.flags[FlagIndex.READY_AOI1] = False
                        self.flags[FlagIndex.READY_AOI0] = True



                    # Process blue1 when ready
                    if self.flags[FlagIndex.RQST_SPK_B1]:
                        self.flags[FlagIndex.RQST_SPK_B1] = False

                        # Detection writes directly to shared memory
                        film_position, edge_position = detect_holes(self.blue1_8, self.sprocket_position_blue1,
                                                    SprocketIndex.POS_FRAME1,
                                                    self.hole_x_start, self.hole_x_end)

                        self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL] = film_position
                        self.sm_uint16var[Uint16Index.FILM_LOCATION_HORIZONTAL] = edge_position - 256 if edge_position < 0xFFFF else 0xFFFF


                        if film_position < 0xFFFF:
                            self.sm_uint16var[Uint16Index.STEPS_TO_MARKER] = int(abs(float(film_position) - float(self.sm_uint16var[Uint16Index.SPROCKET_MARKER])) * float(self.sm_uint16var[Uint16Index.PIX_TO_STEPS]) / 1000.0)
                            self.flags[FlagIndex.TO_MARKER_FWD] = film_position > self.sm_uint16var[Uint16Index.SPROCKET_MARKER]

                        self.flags[FlagIndex.READY_AOI0] = False
                        self.flags[FlagIndex.READY_AOI1] = True


                        
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            log(LogTag.DET, "Interrupted by user")
        except Exception as e:
            log(LogTag.DET, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()
        finally:
            log(LogTag.DET, "Stopping...")


def main():
    """Main function for sprocket detector module"""
    detector = None
    try:
        detector = SprocketDetector()
        detector.process()
    except Exception as e:
        log(LogTag.DET, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.DET, "Shutting down...")
        cleanup_manager()
        log(LogTag.DET, "Cleanup complete")


if __name__ == "__main__":
    main()