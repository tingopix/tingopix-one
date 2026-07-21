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
Sprocket detection (AOI) panel

Displays the area-of-interest view used to verify sprocket-hole detection and
framing. Draws the sprocket marker, the auto-centred crop boundaries, and the
detected hole positions over the AOI image, and shows the film format selector,
the sprocket and crop-height sliders, and a live status readout of position,
hole geometry, exposure, buffer and capture rate.

Shared memory:
    Reads:  READY_AOI0, READY_AOI1, GUI_LOCK_AOI_PANEL, TO_MARKER_FWD,
            MOD_READY_FILESAVER, FILM_LOCATION_VERTICAL, STEPS_TO_MARKER,
            RAW0_EXPOSURE, RAW1_EXPOSURE, FT_CAPTURE_TIME,
            display_aoi_0/1, sprocket_position_blue0/1
    Writes: READY_AOI0, READY_AOI1 (cleared on consume), MOD_READY_GUI_AOI,
            SPROCKET_MARKER, CROP_HEIGHT, AOI_SPROCKET_LOCATION, AOI_POSITION,
            STEPS_PER_FRAME, PIX_TO_STEPS

Interacts with:
    mod_detector    (sprocket hole positions, FILM_LOCATION_VERTICAL)
    mod_sprocket    (READY_AOI0/1 handshake, display_aoi arrays)
    mod_controller  (STEPS_TO_MARKER, TO_MARKER_FWD, STEPS_PER_FRAME)
    mod_raw_capture (RAW0/RAW1_EXPOSURE readout)
    mod_gui         (instantiated as a panel, refreshed on the GUI timer)
    film_config     (film format definitions and crop geometry)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
import numpy as np
from numba import njit
import time


from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex, SprocketIndex, FloatIndex, Uint16Index, Uint32Index
from film_config import get_film_names, get_film_config, get_default_config
from tpx_logger import log, LogTag, log_compile_start, log_compile_done


@njit (cache=True)
def update_display_markers( aoi_array, aoi_marker, sprocket_location, aoi_crop_marker_top, aoi_crop_marker_bottom, aoi_hole0_center, aoi_hole0_top, aoi_hole0_bottom, aoi_hole0_edge,  aoi_hole1_center, aoi_hole1_top, aoi_hole1_bottom, aoi_hole1_edge):
    purple = np.array([192, 0, 192], dtype=np.uint8)
    green = np.array([0, 192, 0], dtype=np.uint8)
    cyan = np.array([0, 192, 192], dtype=np.uint8)
    yellow = np.array([192, 192, 0], dtype=np.uint8)
    gray = np.array([128, 128, 128], dtype=np.uint8)


    marker = aoi_marker // 2
    crop_top = aoi_crop_marker_top // 2 # Auto-centered, divide by 2 for half resolution
    crop_bottom_offset = aoi_crop_marker_bottom // 2
    crop_bottom_pixel = 760 - crop_bottom_offset  # Bottom boundary position

    hole0_center = aoi_hole0_center // 2
    hole0_bottom = aoi_hole0_bottom // 2
    hole0_top = aoi_hole0_top // 2
    hole0_edge = aoi_hole0_edge // 2

    hole1_center = aoi_hole1_center // 2
    hole1_bottom = aoi_hole1_bottom // 2
    hole1_top = aoi_hole1_top // 2
    hole1_edge = aoi_hole1_edge // 2


    # Draw Marker
    aoi_array[0:128, marker-1:marker+1] = gray

    # Draw Crop Markers (purple, 4 pixels wide, Y: 128-192)
    # Crop is always centered vertically
    # AOI display is transposed: shape is (192, 760, 3) - width=192, height=760
    # Sensor height is 1520, display shows half resolution (760)
    
    # Crop Top marker (purple, 4 pixels wide)
    # Clamp to valid range and ensure 4 pixels can be drawn
    crop_top_clamped = max(2, min(crop_top, 758))
    aoi_array[128:192, crop_top_clamped-1:crop_top_clamped+1] = purple
    
    # Crop Bottom marker (purple, 4 pixels wide)
    # Clamp to valid range and ensure 4 pixels can be drawn
    crop_bottom_clamped = max(2, min(crop_bottom_pixel, 758))
    aoi_array[128:192, crop_bottom_clamped-1:crop_bottom_clamped+1] = purple


    # Sprocket Hole 1 - Scaled down by 2 for Presentation
    if hole1_center < 0x7FFF:  # Valid detection
        aoi_array[sprocket_location:sprocket_location+64, hole1_top:hole1_top+2] = cyan
        aoi_array[sprocket_location:sprocket_location+64, hole1_bottom-2:hole1_bottom] = yellow
        aoi_array[sprocket_location:sprocket_location+64, hole1_center-1:hole1_center+1] = green

        if hole1_edge < 0x7FFF:
            aoi_array[hole1_edge-2:hole1_edge, hole1_top:hole1_bottom] = cyan

    # Sprocket Hole 0 - Scaled down by 2 for Presentation
    if hole0_center < 0x7FFF:  # Valid detection
        
        aoi_array[sprocket_location:sprocket_location+64, hole0_bottom-2:hole0_bottom] = yellow
        aoi_array[sprocket_location:sprocket_location+64, hole0_center-1:hole0_center+1] = green

        if hole1_center < 0x7FFF:
            #two holes
            aoi_array[sprocket_location:sprocket_location+64, hole0_top:hole0_top+2] = yellow

        else:
            #single hole
            aoi_array[sprocket_location:sprocket_location+64, hole0_top:hole0_top+2] = cyan

        if hole0_edge < 0x7FFF:
            aoi_array[hole0_edge-2:hole0_edge, hole0_top:hole0_bottom] = cyan




class AOIPanel(BasePanel):
    """Panel showing the AOI view with sprocket-hole and crop overlays."""

    def __init__(self, flags, shared_arrays):
        """Wire up the AOI and sprocket arrays, seed the film configuration, and warm up the marker-drawing JIT."""
        super().__init__("Sprocket Detection", flags, shared_arrays)

        self.display_aoi_0 = shared_arrays['display_aoi_0']
        self.display_aoi_1 = shared_arrays['display_aoi_1']
        self.sprocket_position_blue0 = shared_arrays['sprocket_position_blue0']
        self.sprocket_position_blue1 = shared_arrays['sprocket_position_blue1']
        self.sm_floatvar = shared_arrays['sm_floatvar']
        self.sm_uint16var = shared_arrays['sm_uint16var']
        self.sm_uint32var = shared_arrays['sm_uint32var']

        self.aoi_height, self.aoi_width, self.channels = self.display_aoi_0.shape

        self.picture_aoi = None
        self.sprocket_slider = None
        self.sprocket_value_label = None

        # Film configuration widgets
        self.film_config_combo = None
        self.crop_height_slider = None
        self.crop_height_label = None

        # Individual status labels
        self.status_pos_label = None
        self.status_size_label = None
        self.status_h1_label = None
        self.status_h2_label = None
        self.status_exp_label = None
        self.status_buffer_label = None
        self.status_fps_label = None

        self.sm_uint16var[Uint16Index.SPROCKET_MARKER] = 602
        self.sm_uint16var[Uint16Index.CROP_HEIGHT] = 1200  # Initialize shared memory crop height
        self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] = 20
        self.sm_uint16var[Uint16Index.AOI_POSITION] = 0

        # Crop parameters (class variables for display)
        # crop_height mirrors shared memory, crop_top/bottom are calculated for visualization
        self.crop_top = 160      # Default for Standard 8mm
        self.crop_bottom = 160
        self.crop_height = 1200  # Mirrors shared memory

        log(LogTag.AOI, f"Connected to arrays ({self.aoi_width}x{self.aoi_height})")

        self.config = get_default_config()

        self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] = self.config.sprocket_location
        self.sm_uint16var[Uint16Index.AOI_POSITION] = self.config.aoi_position
        self.sm_uint16var[Uint16Index.SPROCKET_MARKER] = self.config.sprocket_marker_position
        self.sm_uint16var[Uint16Index.STEPS_PER_FRAME] = self.config.stepper_steps_per_frame
        self.sm_uint16var[Uint16Index.PIX_TO_STEPS] = self.config.pix_to_steps

        self.update_markers = False
        self.frame_idx = SprocketIndex.POS_FRAME0
        self.aoi_idx = False
        self.current_sprocket = self.sprocket_position_blue0

        while not self.flags[FlagIndex.MOD_READY_FILESAVER]:
            time.sleep(0.5)

        #Update from shared memory
        self.sprocket_location = self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] // 2
        self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
        self.hole0_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_CENTER]
        self.hole0_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_TOP]
        self.hole0_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_BOTTOM]
        self.hole0_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_EDGE]
        self.hole1_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_CENTER]
        self.hole1_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_TOP]
        self.hole1_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_BOTTOM]
        self.hole1_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_EDGE]
        self.sprocket_marker = self.sm_uint16var[Uint16Index.SPROCKET_MARKER]

        log_compile_start(LogTag.AOI, "display markers function")
        compile_start = time.perf_counter()
        update_display_markers( self.display_aoi_0,
                                self.sprocket_marker,
                                self.sprocket_location,
                                self.crop_top, self.crop_bottom,
                                self.hole0_center,
                                self.hole0_top,
                                self.hole0_bottom,
                                self.hole0_edge,
                                self.hole1_center,
                                self.hole1_top,
                                self.hole1_bottom,
                                self.hole1_edge
                                )
        update_display_markers( self.display_aoi_1,
                                self.sprocket_marker,
                                self.sprocket_location,
                                self.crop_top, self.crop_bottom,
                                self.hole0_center,
                                self.hole0_top,
                                self.hole0_bottom,
                                self.hole0_edge,
                                self.hole1_center,
                                self.hole1_top,
                                self.hole1_bottom,
                                self.hole1_edge
                                )
        compile_time = (time.perf_counter() - compile_start) * 1000
        log_compile_done(LogTag.AOI, compile_time)

        self.flags[FlagIndex.MOD_READY_GUI_AOI] = True

    def create_frame(self):
        """Build the panel: AOI image and sliders on the left, film format and status readout on the right."""
        frame, vbox = self._create_base_frame()

        # Main horizontal box: Left (AOI) | Right (Film Format / Status)
        main_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        # Left frame: AOI image + sliders
        left_frame = Gtk.Frame()
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left_vbox.set_margin_top(4)
        left_vbox.set_margin_bottom(4)
        left_vbox.set_margin_start(4)
        left_vbox.set_margin_end(4)

        # AOI image
        self.picture_aoi = Gtk.Picture()
        self.picture_aoi.set_size_request(self.aoi_width, self.aoi_height)
        self.picture_aoi.set_can_shrink(False)
        left_vbox.append(self.picture_aoi)

        # Sprocket Position slider
        sprocket_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sprocket_label = Gtk.Label(label="Sprocket:")
        sprocket_label.set_xalign(0)
        sprocket_hbox.append(sprocket_label)

        self.sprocket_slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 1, 1520, 1
        )
        self.sprocket_slider.set_value(self.sm_uint16var[Uint16Index.SPROCKET_MARKER])
        self.sprocket_slider.set_draw_value(False)
        self.sprocket_slider.set_hexpand(True)
        self.sprocket_slider.connect("value-changed", self._on_sprocket_position_changed)
        sprocket_hbox.append(self.sprocket_slider)

        self.sprocket_value_label = Gtk.Label(label=str(self.sm_uint16var[Uint16Index.SPROCKET_MARKER]))
        self.sprocket_value_label.set_size_request(40, -1)
        sprocket_hbox.append(self.sprocket_value_label)

        left_vbox.append(sprocket_hbox)

        # Crop Height slider
        crop_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        crop_label = Gtk.Label(label="Crop Height:")
        crop_label.set_xalign(0)
        crop_hbox.append(crop_label)

        self.crop_height_slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 760, 1520, 10
        )
        self.crop_height_slider.set_value(1200)
        self.crop_height_slider.set_draw_value(False)
        self.crop_height_slider.set_hexpand(True)
        self.crop_height_slider.connect("value-changed", self._on_crop_height_changed)
        crop_hbox.append(self.crop_height_slider)

        self.crop_height_label = Gtk.Label(label="1200")
        self.crop_height_label.set_size_request(40, -1)
        crop_hbox.append(self.crop_height_label)

        left_vbox.append(crop_hbox)

        left_frame.set_child(left_vbox)
        main_hbox.append(left_frame)

        # Right frame: Film Format + Status items
        right_frame = Gtk.Frame()
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right_vbox.set_margin_top(4)
        right_vbox.set_margin_bottom(4)
        right_vbox.set_margin_start(4)
        right_vbox.set_margin_end(4)

        # Film Format combo
        format_label = Gtk.Label(label="Film Format:")
        format_label.set_xalign(0)
        right_vbox.append(format_label)

        self.film_config_combo = Gtk.ComboBoxText()
        for film_name in get_film_names():
            self.film_config_combo.append_text(film_name)
        self.film_config_combo.set_active(0)
        self.film_config_combo.connect("changed", self._on_film_config_changed)
        right_vbox.append(self.film_config_combo)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.set_margin_top(4)
        sep.set_margin_bottom(4)
        right_vbox.append(sep)

        # Status labels — individual, stacked vertically
        def make_status_row(title):
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            title_lbl = Gtk.Label(label=title)
            title_lbl.set_xalign(0)
            title_lbl.set_size_request(60, -1)
            hbox.append(title_lbl)
            value_lbl = Gtk.Label(label="----")
            value_lbl.set_xalign(0)
            hbox.append(value_lbl)
            right_vbox.append(hbox)
            return value_lbl

        self.status_pos_label    = make_status_row("Pos:")
        self.status_h1_label     = make_status_row("Hole 1:")
        self.status_h2_label     = make_status_row("Hole 2:")
        self.status_exp_label    = make_status_row("Exp:")
        self.status_size_label   = make_status_row("Size:")
        self.status_buffer_label = make_status_row("Buffer:")
        self.status_fps_label    = make_status_row("Avg. FPS:")

        right_frame.set_child(right_vbox)
        main_hbox.append(right_frame)

        vbox.append(main_hbox)

        # Load default film configuration
        self._on_film_config_changed(self.film_config_combo)

        return frame

    def _on_sprocket_position_changed(self, scale):
        """Write the new sprocket marker position to shared memory and update its label."""
        value = int(scale.get_value())
        self.sm_uint16var[Uint16Index.SPROCKET_MARKER] = value # set position to shared memory
        self.sprocket_value_label.set_text(str(value))

        # TODO: Update sprocket position in shared memory or camera

    def _on_film_config_changed(self, combo):
        """Apply the selected film format: publish its geometry to shared memory and recentre the crop."""
        film_name = combo.get_active_text()
        if film_name:
            try:
                self.config = get_film_config(film_name)

                # Update sprocket marker (still in shared memory - used by other modules)
                self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] = self.config.sprocket_location
                self.sm_uint16var[Uint16Index.AOI_POSITION] = self.config.aoi_position
                self.sm_uint16var[Uint16Index.SPROCKET_MARKER] = self.config.sprocket_marker_position
                self.sm_uint16var[Uint16Index.STEPS_PER_FRAME] = self.config.stepper_steps_per_frame
                self.sm_uint16var[Uint16Index.PIX_TO_STEPS] = self.config.pix_to_steps

                self.sprocket_slider.set_value(self.config.sprocket_marker_position)
                self.sprocket_value_label.set_text(str(self.config.sprocket_marker_position))

                # Update crop height slider (block signal to prevent cascading updates)
                self.crop_height_slider.handler_block_by_func(self._on_crop_height_changed)
                self.crop_height_slider.set_value(self.config.crop_height)
                self.crop_height_label.set_text(str(self.config.crop_height))
                self.crop_height_slider.handler_unblock_by_func(self._on_crop_height_changed)

                # Calculate centered crop_top automatically and update shared memory
                self.crop_height = self.config.crop_height
                self.sm_uint16var[Uint16Index.CROP_HEIGHT] = self.config.crop_height  # Write to shared memory
                self.crop_top = (1520 - self.config.crop_height) // 2
                self.crop_bottom = 1520 - self.crop_top - self.config.crop_height

                log(LogTag.AOI, f"Film config → {film_name}")
                log(LogTag.AOI, f"  Crop: height={self.config.crop_height}, top={self.crop_top} (centered), bottom={self.crop_bottom}")
                log(LogTag.AOI, f"  Sprocket marker: {self.config.sprocket_marker_position}")

            except KeyError as e:
                log(LogTag.AOI, f"Error loading film config — {e}", level="ERR")

    def _on_crop_height_changed(self, scale):
        """Publish the new crop height and recompute the vertically centred crop boundaries."""
        height = int(scale.get_value())

        # Auto-calculate centered crop_top
        self.crop_height = height
        self.sm_uint16var[Uint16Index.CROP_HEIGHT] = height  # Write to shared memory
        self.crop_top = (1520 - height) // 2
        self.crop_bottom = 1520 - self.crop_top - height

        # Update label
        self.crop_height_label.set_text(str(height))

    def _update_widget_sensitivity(self):
        """Disable the interactive controls while the AOI panel is locked."""
        locked = self.flags[FlagIndex.GUI_LOCK_AOI_PANEL]

        # Disable/enable all interactive widgets
        self.sprocket_slider.set_sensitive(not locked)
        self.film_config_combo.set_sensitive(not locked)
        self.crop_height_slider.set_sensitive(not locked)

    def update_display(self):
        """Redraw whichever AOI buffer is ready and refresh the status readout, returning False if the update fails."""
        try:
            # Update widget sensitivity based on lock flag
            self._update_widget_sensitivity()

            if (self.flags[FlagIndex.READY_AOI0]):
                self.flags[FlagIndex.READY_AOI0] = False
                self.aoi_idx = False
                self.current_sprocket = self.sprocket_position_blue0
                self.frame_idx = SprocketIndex.POS_FRAME0

                #Update from shared memory
                self.sprocket_location = self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] // 2
                self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
                self.hole0_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_CENTER]
                self.hole0_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_TOP]
                self.hole0_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_BOTTOM]
                self.hole0_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_EDGE]
                self.hole1_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_CENTER]
                self.hole1_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_TOP]
                self.hole1_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_BOTTOM]
                self.hole1_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_EDGE]
                self.sprocket_marker = self.sm_uint16var[Uint16Index.SPROCKET_MARKER]

                update_display_markers( self.display_aoi_0,
                                       self.sprocket_marker,
                                       self.sprocket_location,
                                       self.crop_top, self.crop_bottom,
                                       self.hole0_center,
                                       self.hole0_top,
                                       self.hole0_bottom,
                                       self.hole0_edge,
                                       self.hole1_center,
                                       self.hole1_top,
                                       self.hole1_bottom,
                                       self.hole1_edge
                                       )

                # Update AOI picture
                stride = self.aoi_width * self.channels
                texture = Gdk.MemoryTexture.new(
                    self.aoi_width, self.aoi_height,
                    Gdk.MemoryFormat.R8G8B8,
                    GLib.Bytes.new(self.display_aoi_0.tobytes()),
                    stride
                )
                self.picture_aoi.set_paintable(texture)

            if (self.flags[FlagIndex.READY_AOI1]):
                self.flags[FlagIndex.READY_AOI1] = False
                self.aoi_idx = True
                self.current_sprocket = self.sprocket_position_blue1

                self.frame_idx = SprocketIndex.POS_FRAME1

                #Update from shared memory
                self.sprocket_location = self.sm_uint16var[Uint16Index.AOI_SPROCKET_LOCATION] // 2
                self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
                self.hole0_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_CENTER]
                self.hole0_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_TOP]
                self.hole0_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_BOTTOM]
                self.hole0_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE0, SprocketIndex.POS_EDGE]
                self.hole1_center = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_CENTER]
                self.hole1_top = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_TOP]
                self.hole1_bottom = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_BOTTOM]
                self.hole1_edge = self.current_sprocket[self.frame_idx, SprocketIndex.POS_HOLE1, SprocketIndex.POS_EDGE]
                self.sprocket_marker = self.sm_uint16var[Uint16Index.SPROCKET_MARKER]

                update_display_markers( self.display_aoi_1,
                                       self.sprocket_marker,
                                       self.sprocket_location,
                                       self.crop_top, self.crop_bottom,
                                       self.hole0_center,
                                       self.hole0_top,
                                       self.hole0_bottom,
                                       self.hole0_edge,
                                       self.hole1_center,
                                       self.hole1_top,
                                       self.hole1_bottom,
                                       self.hole1_edge
                                       )

                # Update AOI picture
                stride = self.aoi_width * self.channels
                texture = Gdk.MemoryTexture.new(
                    self.aoi_width, self.aoi_height,
                    Gdk.MemoryFormat.R8G8B8,
                    GLib.Bytes.new(self.display_aoi_1.tobytes()),
                    stride
                )
                self.picture_aoi.set_paintable(texture)

            # Update individual status labels
            if self.film_vertical < 0xFFFF:
                film_position_difference = self.sm_uint16var[Uint16Index.STEPS_TO_MARKER]
                direction = "<" if self.flags[FlagIndex.TO_MARKER_FWD] else ">"
                self.status_pos_label.set_text(f"{self.film_vertical:04d} {direction} {film_position_difference:04x}")
            else:
                self.status_pos_label.set_text("----")

            self.status_size_label.set_text(f"{self.aoi_width} x {self.aoi_height}")

            if self.hole0_center < 0xFFFF:
                self.status_h1_label.set_text(f"{self.hole0_top:04d} - {self.hole0_bottom:04d}")
            else:
                self.status_h1_label.set_text("----")

            if self.hole1_center < 0xFFFF:
                self.status_h2_label.set_text(f"{self.hole1_top:04d} - {self.hole1_bottom:04d}")
            else:
                self.status_h2_label.set_text("----")

            if self.aoi_idx:
                self.status_exp_label.set_text(f"{self.sm_uint32var[Uint32Index.RAW1_EXPOSURE]:06d}")
                self.status_buffer_label.set_text("B1")
            else:
                self.status_exp_label.set_text(f"{self.sm_uint32var[Uint32Index.RAW0_EXPOSURE]:06d}")
                self.status_buffer_label.set_text("B0")

            self.status_fps_label.set_text(f"{self.sm_floatvar[FloatIndex.FT_CAPTURE_TIME]:.2f}")

            return True

        except Exception as e:
            log(LogTag.AOI, f"Update error — {e}", level="ERR")
            return False