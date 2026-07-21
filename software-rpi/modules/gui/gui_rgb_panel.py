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
Raw monitor panel

Shows the live debayered preview from whichever RGB buffer is ready, and carries
the switches that control how that preview is rendered: colour processing on or
off, inverted output for negative film, a focus aid with its own indicator
overlay, and a live/buffer source selector. The switches only publish flags; the
rendering itself happens in the debayer module.

Shared memory:
    Reads:  READY_RGB0, READY_RGB1, RGB_8_FOCUS_INDICATOR
    Writes: READY_RGB0, READY_RGB1 (cleared on consume), RGB_8_FOCUS,
            RGB_8_FOCUS_INDICATOR, RGB_8_SHOW_BUFFER, RGB_8_COLOR_PROCESSING,
            RGB_8_INVERT

Interacts with:
    mod_debayer (consumes every RGB_8_* switch flag; drives READY_RGB0/1 and
                 fills the rgb0_8 / rgb1_8 display buffers)
    mod_gui     (instantiated as a panel, refreshed on the GUI timer)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex
from tpx_logger import log, LogTag


class RGBPanel(BasePanel):
    """Panel showing the live raw monitor preview and its display switches."""

    def __init__(self, flags, shared_arrays):
        """Bind the two RGB display buffers and set the switch flags to their startup defaults."""
        super().__init__("Raw Monitor", flags, shared_arrays)

        self.rgb0_8 = shared_arrays['rgb0_8']
        self.rgb1_8 = shared_arrays['rgb1_8']

        self.rgb_height, self.rgb_width, self.channels = self.rgb0_8.shape
        self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR] = False
        self.flags[FlagIndex.RGB_8_SHOW_BUFFER] = False
        self.flags[FlagIndex.RGB_8_COLOR_PROCESSING] = True
        self.flags[FlagIndex.RGB_8_INVERT] = False

        self.picture_rgb = None
        self.color_processing_switch = None
        self.live_buffer_switch = None
        self.focus_switch = None
        self.indicator_switch = None
        self.invert_switch = None
        self.rgb_name = "Buffer 0"

        log(LogTag.GUI, f"RGB Panel - Connected to arrays ({self.rgb_width}x{self.rgb_height})")

    def create_frame(self):
        """Build the panel: the preview image above a row of display switches and the buffer status label."""
        frame, vbox = self._create_base_frame()
        frame.set_halign(Gtk.Align.START)

        # Picture widget for RGB display
        self.picture_rgb = Gtk.Picture()
        self.picture_rgb.set_size_request(self.rgb_width, self.rgb_height)
        self.picture_rgb.set_can_shrink(False)
        vbox.append(self.picture_rgb)

        # Horizontal box for switches row (switch labels + status label all on top row)
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        # Switches container (left side)
        switches_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        # Fixed width per switch cell so columns are uniform regardless of label length
        SWITCH_CELL_WIDTH = 80

        # Live/Buffer switch with label
        live_buffer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        live_buffer_box.set_halign(Gtk.Align.CENTER)
        live_buffer_box.set_size_request(SWITCH_CELL_WIDTH, -1)
        live_buffer_label = Gtk.Label(label="Live/Buffer")
        live_buffer_box.append(live_buffer_label)
        self.live_buffer_switch = Gtk.Switch()
        self.live_buffer_switch.set_halign(Gtk.Align.CENTER)
        self.live_buffer_switch.connect("state-set", self._on_live_buffer_toggled)
        live_buffer_box.append(self.live_buffer_switch)
        switches_box.append(live_buffer_box)

        # Raw/Color switch with label - True = Color processing ON
        color_raw_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        color_raw_box.set_halign(Gtk.Align.CENTER)
        color_raw_box.set_size_request(SWITCH_CELL_WIDTH, -1)
        color_raw_label = Gtk.Label(label="Raw/Color")
        color_raw_box.append(color_raw_label)
        self.color_processing_switch = Gtk.Switch()
        self.color_processing_switch.set_halign(Gtk.Align.CENTER)
        self.color_processing_switch.set_active(True)
        self.color_processing_switch.connect("state-set", self._on_color_processing_toggled)
        color_raw_box.append(self.color_processing_switch)
        switches_box.append(color_raw_box)

        # Invert switch with label (negative film preview)
        invert_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        invert_box.set_halign(Gtk.Align.CENTER)
        invert_box.set_size_request(SWITCH_CELL_WIDTH, -1)
        invert_label = Gtk.Label(label="Invert")
        invert_box.append(invert_label)
        self.invert_switch = Gtk.Switch()
        self.invert_switch.set_halign(Gtk.Align.CENTER)
        self.invert_switch.connect("state-set", self._on_invert_toggled)
        invert_box.append(self.invert_switch)
        switches_box.append(invert_box)

        # Focus switch with label
        focus_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        focus_box.set_halign(Gtk.Align.CENTER)
        focus_box.set_size_request(SWITCH_CELL_WIDTH, -1)
        focus_label = Gtk.Label(label="Focus")
        focus_box.append(focus_label)
        self.focus_switch = Gtk.Switch()
        self.focus_switch.set_halign(Gtk.Align.CENTER)
        self.focus_switch.connect("state-set", self._on_focus_toggled)
        focus_box.append(self.focus_switch)
        switches_box.append(focus_box)

        # Indicator switch with label
        indicator_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        indicator_box.set_halign(Gtk.Align.CENTER)
        indicator_box.set_size_request(SWITCH_CELL_WIDTH, -1)
        indicator_label = Gtk.Label(label="Indicator")
        indicator_box.append(indicator_label)
        self.indicator_switch = Gtk.Switch()
        self.indicator_switch.set_halign(Gtk.Align.CENTER)
        self.indicator_switch.set_sensitive(False)  # Start disabled
        self.indicator_switch.connect("state-set", self._on_indicator_toggled)
        indicator_box.append(self.indicator_switch)
        switches_box.append(indicator_box)

        hbox.append(switches_box)

        # Status label (right-justified on same row as switch labels, top-aligned)
        status_label = self._create_status_label("BfrX")
        status_label.set_hexpand(True)
        status_label.set_halign(Gtk.Align.END)
        status_label.set_valign(Gtk.Align.START)
        hbox.append(status_label)

        vbox.append(hbox)

        return frame

    def _on_color_processing_toggled(self, switch, state):
        """Publish whether colour processing is applied to the preview."""
        self.flags[FlagIndex.RGB_8_COLOR_PROCESSING] = state
        log(LogTag.GUI, f"Color processing toggled to {state}")
        return False

    def _on_live_buffer_toggled(self, switch, state):
        """Publish whether the preview shows the live feed or the captured buffer."""
        self.flags[FlagIndex.RGB_8_SHOW_BUFFER] = state
        log(LogTag.GUI, f"Live/Buffer toggled to {state}")
        return False  # Allow the switch to change state

    def _on_focus_toggled(self, switch, state):
        """Publish the focus aid state, and enable or restore the indicator switch to match."""
        self.flags[FlagIndex.RGB_8_FOCUS] = state

        # Enable/disable Indicator switch based on Focus state
        self.indicator_switch.set_sensitive(state)

        if state:
            # Focus turning ON - restore last indicator state from shared memory
            last_indicator_state = self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR]
            # Block the signal handler temporarily to avoid triggering
            self.indicator_switch.handler_block_by_func(self._on_indicator_toggled)
            self.indicator_switch.set_active(last_indicator_state)
            self.indicator_switch.handler_unblock_by_func(self._on_indicator_toggled)
        else:
            # Focus turning OFF - turn off indicator
            if self.indicator_switch.get_active():
                self.indicator_switch.handler_block_by_func(self._on_indicator_toggled)
                self.indicator_switch.set_active(False)
                self.indicator_switch.handler_unblock_by_func(self._on_indicator_toggled)

        log(LogTag.GUI, f"RGB_8_FOCUS toggled to {state}")
        # No explicit return: falling off the end yields None, which GTK treats
        # as False and lets the switch change state, as in the other handlers.

    def _on_indicator_toggled(self, switch, state):
        """Publish whether the focus indicator overlay is shown."""
        self.flags[FlagIndex.RGB_8_FOCUS_INDICATOR] = state
        log(LogTag.GUI, f"Indicator toggled to {state}")
        return False  # Allow the switch to change state

    def _on_invert_toggled(self, switch, state):
        """Publish whether the preview is inverted for negative film."""
        self.flags[FlagIndex.RGB_8_INVERT] = state
        log(LogTag.GUI, f"RGB_8_INVERT toggled to {state}")
        return False  # Allow the switch to change state

    def update_display(self):
        """Redraw whichever RGB buffer is ready and update the buffer label, returning False if the update fails."""
        try:
            if (self.flags[FlagIndex.READY_RGB0]):
                self.flags[FlagIndex.READY_RGB0] = False
                current_rgb = self.rgb0_8
                self.rgb_name = "Bfr0"

                stride = self.rgb_width * self.channels
                texture = Gdk.MemoryTexture.new(
                    self.rgb_width, self.rgb_height,
                    Gdk.MemoryFormat.R8G8B8,
                    GLib.Bytes.new(current_rgb.tobytes()),
                    stride
                )
                self.picture_rgb.set_paintable(texture)

                self.update_status(
                    f"{self.rgb_name}"
                )

            if (self.flags[FlagIndex.READY_RGB1]):
                self.flags[FlagIndex.READY_RGB1] = False
                current_rgb = self.rgb1_8
                self.rgb_name = "Bfr1"

                stride = self.rgb_width * self.channels
                texture = Gdk.MemoryTexture.new(
                    self.rgb_width, self.rgb_height,
                    Gdk.MemoryFormat.R8G8B8,
                    GLib.Bytes.new(current_rgb.tobytes()),
                    stride
                )
                self.picture_rgb.set_paintable(texture)

                self.update_status(
                    f"{self.rgb_name}"
                )

            return True

        except Exception as e:
            log(LogTag.GUI, f"RGB Panel update error — {e}", level="ERR")
            return False