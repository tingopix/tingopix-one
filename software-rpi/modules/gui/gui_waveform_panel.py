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
Raw waveform panel

Draws the RGB waveform used to judge exposure. A cached graticule of labelled
level lines is rendered underneath, with the waveform painted over it from
whichever shared buffer the waveform module has most recently filled. Also
carries the parade switch that selects between overlaid and side-by-side
channel display.

The waveform module sets READY_WVF0 or READY_WVF1 to mark whichever buffer holds
the newest waveform, clearing the other. This panel consumes that flag when it
paints, so a redraw happens once per new waveform rather than on every GUI tick.

Shared memory:
    Reads:  READY_WVF0, READY_WVF1, wvf0_8, wvf1_8
    Writes: WVF_PARADE, READY_WVF0 and READY_WVF1 (cleared on consume)

Interacts with:
    mod_waveform (fills wvf0_8 / wvf1_8, owns the READY_WVF0/1 pair,
                  consumes WVF_PARADE)
    mod_gui      (instantiated as a panel, refreshed on the GUI timer)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk
import cairo
import numpy as np

from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex
from tpx_logger import log, LogTag


class WaveformPanel(BasePanel):
    """Panel showing the RGB waveform over a cached level graticule."""

    def __init__(self, flags, shared_arrays):
        """Bind the two waveform buffers and build the cached graticule surface."""
        super().__init__("Raw Waveform", flags, shared_arrays)

        self.wvf0_8 = shared_arrays['wvf0_8']
        self.wvf1_8 = shared_arrays['wvf1_8']

        self.wvf_height, self.wvf_width, _ = self.wvf0_8.shape

        self.waveform_area = None
        self.parade_switch = None
        self.graticule_surface = None

        log(LogTag.GUI, f"Waveform Panel - Connected to arrays ({self.wvf_width}x{self.wvf_height})")

        # Create graticule surface
        self._create_graticule_surface()

    def _create_graticule_surface(self):
        """Render the level lines and their labels once into a reusable surface."""
        log(LogTag.GUI, "Creating graticule surface...")

        self.graticule_surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32,
            self.wvf_width,
            self.wvf_height
        )
        ctx = cairo.Context(self.graticule_surface)

        # Clear to fully transparent
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.paint()

        # Draw horizontal grid lines with labels
        marker = 0
        for gy in range(268, 11, -32):
            # Draw horizontal line (gray: 64,64,64)
            ctx.set_source_rgb(64/255, 64/255, 64/255)
            ctx.set_line_width(1)
            ctx.move_to(0, gy + 0.5)
            ctx.line_to(self.wvf_width - 1, gy + 0.5)
            ctx.stroke()

            # Draw text label
            ctx.set_source_rgb(160/255, 160/255, 0)
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(10)

            text = str(marker)
            ctx.move_to(4, gy - 2)
            ctx.show_text(text)

            marker += 128

        log(LogTag.GUI, "Graticule surface created")

    def create_frame(self):
        """Build the panel: the waveform drawing area above the parade switch and status label."""
        frame, vbox = self._create_base_frame()
        frame.set_halign(Gtk.Align.START)

        # Create the waveform drawing area
        self.waveform_area = Gtk.DrawingArea()
        self.waveform_area.set_content_width(self.wvf_width)
        self.waveform_area.set_content_height(self.wvf_height)
        self.waveform_area.set_draw_func(self._on_draw_waveform)
        vbox.append(self.waveform_area)

        # Horizontal box for switch and status label
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        # Parade switch with label (left side)
        parade_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        parade_label = Gtk.Label(label="Waveform")
        parade_box.append(parade_label)
        self.parade_switch = Gtk.Switch()
        self.parade_switch.connect("state-set", self._on_parade_toggled)
        parade_box.append(self.parade_switch)
        hbox.append(parade_box)

        # Status label (right side, expands to fill space)
        status_label = self._create_status_label("Parade")
        status_label.set_hexpand(True)
        hbox.append(status_label)

        vbox.append(hbox)

        return frame

    def _on_parade_toggled(self, switch, state):
        """Publish whether the waveform is drawn as a parade."""
        self.flags[FlagIndex.WVF_PARADE] = state
        log(LogTag.GUI, f"WVF_PARADE toggled to {state}")
        return False  # Allow the switch to change state

    def _on_draw_waveform(self, area, ctx, width, height):
        """Paint the graticule and the current waveform buffer, falling back to a black or red field on error."""
        try:
            # READY_WVF0/1 mark whichever buffer matches the most recent capture.
            # mod_waveform sets one and clears the other on every new waveform;
            # consuming the flag here means one redraw per waveform, not per tick.
            wvf_buffer = None

            if self.flags[FlagIndex.READY_WVF0]:
                self.flags[FlagIndex.READY_WVF0] = False
                # Make sure buffer is contiguous for Cairo
                wvf_buffer = np.ascontiguousarray(self.wvf0_8)

            elif self.flags[FlagIndex.READY_WVF1]:
                self.flags[FlagIndex.READY_WVF1] = False
                # Make sure buffer is contiguous for Cairo
                wvf_buffer = np.ascontiguousarray(self.wvf1_8)

            # No waveform ready this pass - draw black with graticule
            if wvf_buffer is None or not np.any(wvf_buffer):

                ctx.set_source_rgb(0, 0, 0)
                ctx.rectangle(0, 0, width, height)
                ctx.fill()

                # Still draw graticule even with no waveform data
                if self.graticule_surface:
                    ctx.set_operator(cairo.OPERATOR_OVER)
                    ctx.set_source_surface(self.graticule_surface, 0, 0)
                    ctx.paint_with_alpha(0.9)
                return

            # Draw graticule FIRST (underneath)
            if self.graticule_surface:
                ctx.set_source_surface(self.graticule_surface, 0, 0)
                ctx.paint()

            # Create Cairo surface from shared memory
            waveform_surface = cairo.ImageSurface.create_for_data(
                wvf_buffer.data,
                cairo.FORMAT_ARGB32,
                self.wvf_width,
                self.wvf_height,
                self.wvf_width * 4
            )

            # Draw waveform on top with transparency so graticule shows through
            ctx.set_source_surface(waveform_surface, 0, 0)
            ctx.paint()

        except Exception as e:
            log(LogTag.GUI, f"Waveform draw error — {e}", level="ERR")
            # Draw error state - red background
            ctx.set_source_rgb(0.3, 0, 0)
            ctx.rectangle(0, 0, width, height)
            ctx.fill()

    def update_display(self):
        """Queue a redraw when the waveform module has published a new buffer, returning False if it fails."""
        try:
            # Redraw only when mod_waveform has published a new waveform;
            # the draw callback clears the flag as it consumes the buffer.
            if self.flags[FlagIndex.READY_WVF0] or self.flags[FlagIndex.READY_WVF1]:
                self.waveform_area.queue_draw()

            return True

        except Exception as e:
            log(LogTag.GUI, f"Waveform Panel update error — {e}", level="ERR")
            return False