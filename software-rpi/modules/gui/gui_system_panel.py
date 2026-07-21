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
System monitor panel

Displays the project logo and a live readout of per-core CPU load and RAM usage,
polled directly from psutil on the GUI refresh timer. This panel is purely
diagnostic: it participates in no shared-memory handshakes and reads nothing the
rest of the pipeline produces.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    mod_gui (instantiated as a panel, refreshed on the GUI timer). The panel
    receives the flags array through the base class but never reads or writes
    it; all data comes from psutil.
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Pango, Gdk
from pathlib import Path

import psutil

from modules.gui.gui_base_panel import BasePanel
from tpx_logger import log, LogTag


LOGO_PATH = Path(__file__).parent.parent.parent / "tpx.png"


class SystemPanel(BasePanel):
    """Panel showing the logo alongside per-core CPU load and RAM usage."""

    def __init__(self, flags, shared_arrays):
        """Prepare the label lists and take a first psutil reading so later samples are meaningful."""
        super().__init__("System", flags, shared_arrays)

        self.cpu_labels = []
        self.ram_label = None

        # Prime psutil so first call returns valid data
        psutil.cpu_percent(interval=None, percpu=True)
        psutil.virtual_memory()

        log(LogTag.GUI, "System Panel - Initialized")

    def create_frame(self):
        """Build the panel: the logo above one row per CPU core and a RAM row."""
        frame, vbox = self._create_base_frame()
        frame.set_size_request(110, -1)

        # Logo at top, scaled to fit panel width
        picture = Gtk.Picture()
        picture.set_halign(Gtk.Align.CENTER)
        picture.set_can_shrink(True)
        picture.set_hexpand(True)
        try:
            texture = Gdk.Texture.new_from_filename(str(LOGO_PATH))
            picture.set_paintable(texture)
        except Exception as e:
            log(LogTag.GUI, f"Logo not found — {e}", level="WARN")
        vbox.append(picture)

        sep_logo = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep_logo.set_margin_top(4)
        sep_logo.set_margin_bottom(4)
        vbox.append(sep_logo)

        num_cores = psutil.cpu_count(logical=True)

        def make_row(title):
            """Add one titled row to the panel and return its value label."""
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            hbox.set_halign(Gtk.Align.CENTER)
            title_lbl = Gtk.Label(label=title)
            title_lbl.set_xalign(1)
            title_lbl.set_size_request(52, -1)
            hbox.append(title_lbl)
            value_lbl = Gtk.Label(label="-- %")
            value_lbl.set_xalign(0)
            value_lbl.set_size_request(44, -1)
            attrs = Pango.AttrList()
            attrs.insert(Pango.attr_family_new("Monospace"))
            value_lbl.set_attributes(attrs)
            hbox.append(value_lbl)
            vbox.append(hbox)
            return value_lbl

        for i in range(num_cores):
            lbl = make_row(f"Core {i}:")
            self.cpu_labels.append(lbl)

        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.set_margin_top(4)
        sep.set_margin_bottom(4)
        vbox.append(sep)

        self.ram_label = make_row("RAM:")

        return frame

    def update_display(self):
        """Refresh the per-core CPU and RAM readouts from psutil."""
        try:
            core_percents = psutil.cpu_percent(interval=None, percpu=True)
            for i, lbl in enumerate(self.cpu_labels):
                if i < len(core_percents):
                    lbl.set_text(f"{core_percents[i]:3.0f} %")

            ram = psutil.virtual_memory()
            self.ram_label.set_text(f"{ram.percent:3.0f} %")

            return True
        except Exception as e:
            log(LogTag.GUI, f"System panel update error — {e}", level="ERR")
            return True