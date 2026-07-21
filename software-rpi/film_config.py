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
Film format configuration.

Defines scanning parameters for each supported film format (crop geometry,
sprocket detection, stepper motion, and physical dimensions) and provides
lookup and validation helpers.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -
"""

from typing import Dict, NamedTuple


class FilmConfig(NamedTuple):
    """Configuration parameters for a film format"""
    name: str

    # Cropping parameters (in pixels)
    crop_height: int  # Height of the film frame (1520 max)
    crop_top: int     # Top offset (0-760 range)
    sprocket_location: int
    aoi_position: int

    # Sprocket parameters
    sprocket_marker_position: int  # Default sprocket marker position (1-1520)
    sprocket_hole_x_start: int     # X start for hole detection
    sprocket_hole_x_end: int       # X end for hole detection

    # Stepper motor parameters
    stepper_steps_per_frame: int   # Number of steps to advance one frame
    pix_to_steps: int   # Number of steps to advance one frame
    stepper_speed: int             # Speed setting for stepper motor

    # Film dimensions (informational)
    film_width_mm: float           # Physical film width in mm
    frame_height_mm: float         # Physical frame height in mm

# IMPORTANT: AOI POSITION IS THE ACTUAL RAW PIXEL, AND IT MUST BE AN ODD NUMBER TO CORRESPOND TO BLUE CHANNEL.

# Film format configurations
FILM_CONFIGS: Dict[str, FilmConfig] = {
    "Standard 8mm": FilmConfig(
        name="Standard 8mm",
        crop_height=1300,
        crop_top=160,
        sprocket_location=100,
        aoi_position = 1, # ODD NUMBER
        sprocket_marker_position=1100,
        sprocket_hole_x_start=5,
        sprocket_hole_x_end=105,
        stepper_steps_per_frame=1350,
        pix_to_steps = 1300,
        stepper_speed=100,
        film_width_mm=8.0,
        frame_height_mm=3.68
    ),

    "Shrunk 8mm": FilmConfig(
        name="Standard 8mm",
        crop_height=1300,
        crop_top=160,
        sprocket_location=100,
        aoi_position = 101, # ODD NUMBER
        sprocket_marker_position=1100,
        sprocket_hole_x_start=5,
        sprocket_hole_x_end=105,
        stepper_steps_per_frame=1350,
        pix_to_steps = 1300,
        stepper_speed=100,
        film_width_mm=8.0,
        frame_height_mm=3.68
    ),

    "Super 8mm": FilmConfig(
        name="Super 8mm",
        crop_height=1400,
        crop_top=60,
        sprocket_location=5,
        aoi_position = 1, # ODD NUMBER
        sprocket_marker_position=600,
        sprocket_hole_x_start=10,
        sprocket_hole_x_end=120,
        stepper_steps_per_frame=1512,
        pix_to_steps = 1300,
        stepper_speed=120,
        film_width_mm=8.0,
        frame_height_mm=5.36
    ),

    "16mm L Perf": FilmConfig(
        name="16mm L",
        crop_height=1200,
        crop_top=0,
        sprocket_location=100,
        aoi_position = 1,
        sprocket_marker_position=1170,
        sprocket_hole_x_start=5,
        sprocket_hole_x_end=132,
        stepper_steps_per_frame=2561,
        pix_to_steps = 2673,
        stepper_speed=150,
        film_width_mm=16.0,
        frame_height_mm=7.49
    ),

    "16mm ZoomS R Perf": FilmConfig(
        name="16mm R",
        crop_height=1300,
        crop_top=0,
        sprocket_location=5,
        aoi_position = 3295, #384 x 2 = 768. 4064-768 = 3296 -> 3295 ODD
        sprocket_marker_position=1250,
        sprocket_hole_x_start=5,
        sprocket_hole_x_end=132,
        stepper_steps_per_frame=2561,
        pix_to_steps = 2193,
        stepper_speed=150,
        film_width_mm=16.0,
        frame_height_mm=7.49
    ),
}


def get_film_config(name: str) -> FilmConfig:
    """
    Get film configuration by name.

    Args:
        name: Film format name

    Returns:
        FilmConfig for the specified format

    Raises:
        KeyError if film format not found
    """
    if name not in FILM_CONFIGS:
        raise KeyError(f"Film format '{name}' not found. Available formats: {list(FILM_CONFIGS.keys())}")
    return FILM_CONFIGS[name]


def get_film_names() -> list:
    """Get list of available film format names"""
    return list(FILM_CONFIGS.keys())


def validate_crop_parameters(height: int, top: int) -> tuple:
    """
    Validate and clamp crop parameters to valid ranges.

    Args:
        height: Crop height (760-1520)
        top: Top offset (0-760)

    Returns:
        Tuple of (clamped_height, clamped_top)
    """
    # Clamp height to valid range
    height = max(760, min(1520, height))

    # Clamp top to valid range
    top = max(0, min(760, top))

    # Ensure top + height doesn't exceed sensor height
    if top + height > 1520:
        top = 1520 - height

    return height, top


def get_default_config() -> FilmConfig:
    """Get the default film configuration (Standard 8mm)"""
    return FILM_CONFIGS["Standard 8mm"]