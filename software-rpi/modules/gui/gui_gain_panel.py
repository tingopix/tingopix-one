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
Digital gain panel

Provides the red and blue digital gain controls applied during debayering. Both
values live in shared memory as floats and are seeded from the tpx_config
defaults if nothing has set them yet. A Default button restores both to the
configured values, and the spinbuttons resync from shared memory so that loading
an exposure preset is reflected in the display.

Shared memory:
    Reads:  R_DIG_GAIN, B_DIG_GAIN
    Writes: R_DIG_GAIN, B_DIG_GAIN

Interacts with:
    mod_debayer      (consumes both gains in the debayer pipeline)
    mod_raw_capture  (embeds the gains in the capture metadata)
    mod_file_saver   (applies the gains on the save path)
    mod_exp_defaults (preset load overwrites both values)
    mod_gui          (instantiated as a panel, refreshed on the GUI timer)
    tpx_config       (DEFAULT_R_DIG_GAIN, DEFAULT_B_DIG_GAIN)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FloatIndex
from tpx_config import DEFAULT_R_DIG_GAIN, DEFAULT_B_DIG_GAIN
from tpx_logger import log, LogTag

GAIN_MIN  = 0.0
GAIN_MAX  = 5.0
GAIN_STEP = 0.005
GAIN_PAGE = 0.1


class GainPanel(BasePanel):
    """Panel for the red and blue digital gain applied during debayering."""

    def __init__(self, flags, shared_arrays):
        """Bind the shared float array and seed the gains from config if they are still unset."""
        super().__init__("Digital Gain", flags, shared_arrays)

        self.float_var = shared_arrays['sm_floatvar']

        # Seed shared memory on construction
        if self.float_var[FloatIndex.R_DIG_GAIN] == 0.0:
            self.float_var[FloatIndex.R_DIG_GAIN] = DEFAULT_R_DIG_GAIN
        if self.float_var[FloatIndex.B_DIG_GAIN] == 0.0:
            self.float_var[FloatIndex.B_DIG_GAIN] = DEFAULT_B_DIG_GAIN

        self.r_adj = None
        self.b_adj = None
        self._updating = False

        log(LogTag.GUI, "Gain Panel - Initialized")

    def create_frame(self):
        """Compact horizontal row: R label + spinbutton, B label + spinbutton, Default button."""
        frame, vbox = self._create_base_frame()

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.set_halign(Gtk.Align.CENTER)
        hbox.set_margin_top(2)
        hbox.set_margin_bottom(2)

        # R channel
        r_label = Gtk.Label(label="R")
        hbox.append(r_label)

        self.r_adj = Gtk.Adjustment(
            value=float(self.float_var[FloatIndex.R_DIG_GAIN]),
            lower=GAIN_MIN, upper=GAIN_MAX,
            step_increment=GAIN_STEP, page_increment=GAIN_PAGE, page_size=0,
        )
        r_spin = Gtk.SpinButton(adjustment=self.r_adj, climb_rate=GAIN_STEP, digits=3)
        r_spin.set_numeric(True)
        self.r_adj.connect("value-changed", self._on_r_changed)
        hbox.append(r_spin)

        # Spacer
        hbox.append(Gtk.Label(label="  "))

        # B channel
        b_label = Gtk.Label(label="B")
        hbox.append(b_label)

        self.b_adj = Gtk.Adjustment(
            value=float(self.float_var[FloatIndex.B_DIG_GAIN]),
            lower=GAIN_MIN, upper=GAIN_MAX,
            step_increment=GAIN_STEP, page_increment=GAIN_PAGE, page_size=0,
        )
        b_spin = Gtk.SpinButton(adjustment=self.b_adj, climb_rate=GAIN_STEP, digits=3)
        b_spin.set_numeric(True)
        self.b_adj.connect("value-changed", self._on_b_changed)
        hbox.append(b_spin)

        # Spacer
        hbox.append(Gtk.Label(label="  "))

        # Default button
        default_btn = Gtk.Button(label="Default")
        default_btn.connect("clicked", self._on_default_clicked)
        hbox.append(default_btn)

        vbox.append(hbox)
        return frame

    def _on_r_changed(self, adj):
        """Publish the red gain, unless the spinbutton is being updated programmatically."""
        if self._updating:
            return
        self.float_var[FloatIndex.R_DIG_GAIN] = adj.get_value()

    def _on_b_changed(self, adj):
        """Publish the blue gain, unless the spinbutton is being updated programmatically."""
        if self._updating:
            return
        self.float_var[FloatIndex.B_DIG_GAIN] = adj.get_value()

    def _on_default_clicked(self, button):
        """Restore both gains to the configured defaults, in the widgets and in shared memory."""
        self._updating = True
        self.r_adj.set_value(DEFAULT_R_DIG_GAIN)
        self.b_adj.set_value(DEFAULT_B_DIG_GAIN)
        self.float_var[FloatIndex.R_DIG_GAIN] = DEFAULT_R_DIG_GAIN
        self.float_var[FloatIndex.B_DIG_GAIN] = DEFAULT_B_DIG_GAIN
        self._updating = False
        log(LogTag.GUI, f"Gain reset to defaults R={DEFAULT_R_DIG_GAIN:.2f} B={DEFAULT_B_DIG_GAIN:.2f}")

    def update_display(self):
        """Sync spinbuttons from shared memory (e.g. after preset load)."""
        try:
            shm_r = float(self.float_var[FloatIndex.R_DIG_GAIN])
            shm_b = float(self.float_var[FloatIndex.B_DIG_GAIN])
            self._updating = True
            if abs(self.r_adj.get_value() - shm_r) > 1e-4:
                self.r_adj.set_value(shm_r)
            if abs(self.b_adj.get_value() - shm_b) > 1e-4:
                self.b_adj.set_value(shm_b)
            self._updating = False
            return True
        except Exception as e:
            log(LogTag.GUI, f"Gain Panel update error — {e}", level="ERR")
            self._updating = False
            return False