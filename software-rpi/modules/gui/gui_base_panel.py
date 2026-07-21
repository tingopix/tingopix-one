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
GUI panel base class

Defines the common interface and shared widget helpers that all GUI panels
inherit. Holds the panel title, the shared flags array, and the dictionary of
shared-memory arrays a panel needs, and provides standard frame and status-label
constructors so the panels look consistent. Subclasses implement create_frame()
and update_display(); this class touches no shared-memory flags itself.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    Library-style base class, imported directly by the GUI panels. It stores the
    flags and array references passed to it but never reads or writes them.
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk


class BasePanel:
    """Base class for all GUI display panels."""

    def __init__(self, title, flags, shared_arrays):
        """Store the panel title, the shared flags array, and the shared-memory arrays this panel needs."""
        self.title = title
        self.flags = flags
        self.shared_arrays = shared_arrays
        self.frame = None
        self.status_label = None

    def create_frame(self):
        """Build the GTK frame holding this panel's widgets. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement create_frame()")

    def update_display(self):
        """Refresh the panel from shared memory. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement update_display()")

    def update_status(self, status_text):
        """Set the status label's text, if this panel created one."""
        if self.status_label:
            self.status_label.set_text(status_text)

    def cleanup(self):
        """Release any resources the panel holds. Subclasses override this if they need it."""
        pass

    def _create_base_frame(self):
        """Build the standard titled frame and its vertical box, and remember the frame.

        Returns both so subclasses can populate the box.
        """
        frame = Gtk.Frame(label=self.title)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(6)
        vbox.set_margin_bottom(6)
        vbox.set_margin_start(6)
        vbox.set_margin_end(6)
        frame.set_child(vbox)

        self.frame = frame
        return frame, vbox

    def _create_status_label(self, initial_text="Initializing..."):
        """Build the standard left-aligned status label and store it for update_status()."""
        self.status_label = Gtk.Label(label=initial_text)
        self.status_label.set_xalign(0)
        return self.status_label