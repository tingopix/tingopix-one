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
GUI process

Owns the GTK4 main window and the GUI process's main loop. Builds the
three-column layout from the individual panel classes, drives their
per-panel refresh on a 100 ms GLib timer, and signals shutdown when the
window is closed. Panel-level shared-memory traffic lives in the panels
themselves; this module only handles process-level handshake flags.

Shared memory:
    Reads:  MOD_READY_ALL
    Writes: GUI_READY, SHUTDOWN_REQUESTED

Interacts with:
    mp_setup       (spawned via main())
    modules.gui.*  (instantiates all panels, passes them shared arrays)
    shm_manager    (arrays, cleanup_manager)
    all backend modules (GUI_READY / MOD_READY_ALL startup handshake,
                         SHUTDOWN_REQUESTED on close)
"""

import os
os.environ['GTK_A11Y'] = 'none'
os.environ['NO_AT_BRIDGE'] = '1'

import time
import warnings
import gi

# Suppress GLib/GDK warnings on systems with limited compositor support
warnings.filterwarnings('ignore', category=Warning, module='gi')

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib

from tpx_logger import log, LogTag
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex

from modules.gui import (AOIPanel, RGBPanel, WaveformPanel,
                         FileSavePanel, ScanControlPanel, ScanButtonsPanel,
                         GainPanel, SystemPanel, apply_dark_theme)


class GUIProcessor:
    """GTK4 main window orchestrator: owns the window, the panels, and the main loop."""

    def __init__(self):
        log(LogTag.GUI, "Initializing...")

        self.flags = get_array("flags")

        # AOI arrays
        display_aoi_0 = get_array("shm_display_aoi_0")
        display_aoi_1 = get_array("shm_display_aoi_1")
        sprocket_position_blue0 = get_array("shm_sprocket_position_blue0")
        sprocket_position_blue1 = get_array("shm_sprocket_position_blue1")
        sm_floatvar = get_array("shm_float_var")
        sm_uint16var = get_array("shm_uint16_var")
        sm_uint32var = get_array("shm_uint32_var")

        # RGB arrays
        rgb0_8 = get_array("shm_rgb0_8")
        rgb1_8 = get_array("shm_rgb1_8")

        # Waveform arrays
        wvf0_8 = get_array("shm_wvf0_8")
        wvf1_8 = get_array("shm_wvf1_8")

        if (display_aoi_0 is None or display_aoi_1 is None or
            rgb0_8 is None or rgb1_8 is None or
            wvf0_8 is None or wvf1_8 is None or
            sm_uint32var is None or self.flags is None):
            raise RuntimeError("GUI: Failed to get shared arrays")

        # Initialize display panels with their required arrays
        self.aoi_panel = AOIPanel(
            self.flags,
            {
                'display_aoi_0': display_aoi_0,
                'display_aoi_1': display_aoi_1,
                'sprocket_position_blue0': sprocket_position_blue0,
                'sprocket_position_blue1': sprocket_position_blue1,
                'sm_floatvar': sm_floatvar,
                'sm_uint16var': sm_uint16var,
                'sm_uint32var': sm_uint32var,
            }
        )

        self.rgb_panel = RGBPanel(
            self.flags,
            {
                'rgb0_8': rgb0_8,
                'rgb1_8': rgb1_8
            }
        )

        self.waveform_panel = WaveformPanel(
            self.flags,
            {
                'wvf0_8': wvf0_8,
                'wvf1_8': wvf1_8
            }
        )

        self.scancontrol_panel = ScanControlPanel(
            self.flags,
            {
                'sm_uint16var': sm_uint16var,
                'sm_uint32var': sm_uint32var,
                'sm_floatvar':  sm_floatvar
            }
        )

        self.scanbuttons_panel = ScanButtonsPanel(
            self.flags,
            {
                'sm_uint16var': sm_uint16var,
            }
        )

        # Initialize control panels
        self.filesave_panel = FileSavePanel(self.flags, {})
        self.system_panel = SystemPanel(self.flags, {})

        # Gain panel
        self.gain_panel = GainPanel(
            self.flags,
            {
                'sm_floatvar': sm_floatvar,
            }
        )

        self.window = None
        self.display_active = False
        self.main_loop = None

        log(LogTag.GUI, "All panels initialized")

    def create_window(self):
        """Build the main window and assemble all panels into the three-column layout."""
        log(LogTag.GUI, "Creating window...")

        apply_dark_theme()

        self.window = Gtk.Window()
        self.window.set_title("tingopix")

        # Set application name to avoid GTK warnings
        GLib.set_application_name("tingopix")
        GLib.set_prgname("tingopix")

        # Calculate window size based on panel dimensions
        aoi_width = self.aoi_panel.aoi_width
        aoi_height = self.aoi_panel.aoi_height
        rgb_width = self.rgb_panel.rgb_width
        rgb_height = self.rgb_panel.rgb_height
        wvf_width = self.waveform_panel.wvf_width
        wvf_height = self.waveform_panel.wvf_height

        # Adjust window size to accommodate all panels including scan buttons
        window_width = max(aoi_width, rgb_width + wvf_width + 24) + 180  # Extra width for scan buttons panel
        window_height = aoi_height + max(rgb_height, wvf_height) + 350  # Extra space for bottom panels
        self.window.set_default_size(window_width, window_height)
        self.window.set_resizable(False)

        # Main container - horizontal (three columns)
        main_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        main_hbox.set_margin_top(12)
        main_hbox.set_margin_bottom(12)
        main_hbox.set_margin_start(12)
        main_hbox.set_margin_end(12)

        # Left Column: RGB on top, Gain frame, Waveform below
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        rgb_frame = self.rgb_panel.create_frame()
        left_vbox.append(rgb_frame)

        gain_frame = self.gain_panel.create_frame()
        left_vbox.append(gain_frame)

        wvf_frame = self.waveform_panel.create_frame()
        left_vbox.append(wvf_frame)

        main_hbox.append(left_vbox)

        # Middle Column: AOI, Scan Control, File Save stacked
        middle_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        aoi_frame = self.aoi_panel.create_frame()
        middle_vbox.append(aoi_frame)

        scancontrol_frame = self.scancontrol_panel.create_frame()
        middle_vbox.append(scancontrol_frame)

        filesave_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        filesave_frame = self.filesave_panel.create_frame()
        filesave_hbox.append(filesave_frame)
        system_frame = self.system_panel.create_frame()
        filesave_hbox.append(system_frame)
        middle_vbox.append(filesave_hbox)

        main_hbox.append(middle_vbox)

        # Right Column: Scan Buttons
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)

        scanbuttons_frame = self.scanbuttons_panel.create_frame()
        right_vbox.append(scanbuttons_frame)

        main_hbox.append(right_vbox)

        self.window.set_child(main_hbox)

        self.window.connect("close-request", self.on_window_close)

        # Show window - use present() for better window manager compatibility
        self.window.present()
        log(LogTag.GUI, "Window visible")

    def update_display(self):
        """Refresh every panel, returning False to cancel the GLib timer once the display is inactive."""
        if not self.display_active:
            return False

        # Update display panels (control panels don't need updates)
        self.aoi_panel.update_display()
        self.system_panel.update_display()
        self.rgb_panel.update_display()
        self.waveform_panel.update_display()

        # Update filesave panel to show incremented frame number
        self.filesave_panel.update_display()

        # Scan buttons panel updates via its own timer
        self.scanbuttons_panel.update_display()

        # Sync gain spinbuttons after preset load
        self.gain_panel.update_display()

        return True

    def on_window_close(self, window):
        """Signal shutdown, clean up all panels, and stop the main loop."""
        log(LogTag.GUI, "Window close requested")
        self.display_active = False

        # Set shutdown flag for other processes
        if self.flags is not None:
            self.flags[FlagIndex.SHUTDOWN_REQUESTED] = True

        self.aoi_panel.cleanup()
        self.rgb_panel.cleanup()
        self.waveform_panel.cleanup()
        self.filesave_panel.cleanup()
        self.scancontrol_panel.cleanup()
        self.scanbuttons_panel.cleanup()
        self.gain_panel.cleanup()

        if self.main_loop and self.main_loop.is_running():
            self.main_loop.quit()
        return False

    def run(self):
        """Wait for the backend modules to finish compiling, then open the window and run the GTK main loop."""
        log(LogTag.GUI, "Starting...")

        self.flags[FlagIndex.GUI_READY] = True

        # Wait until all backend modules have finished compiling
        log(LogTag.GUI, "Waiting for modules to compile...")
        while not self.flags[FlagIndex.MOD_READY_ALL]:
            time.sleep(0.5)
        log(LogTag.GUI, "All modules ready — opening window")

        try:
            self.create_window()
            self.display_active = True

            # Update display every 100ms
            GLib.timeout_add(100, self.update_display)

            log(LogTag.GUI, "Starting main loop...")
            self.main_loop = GLib.MainLoop()
            self.main_loop.run()
            log(LogTag.GUI, "Main loop ended")

        except Exception as e:
            log(LogTag.GUI, f"Error — {e}", level="ERR")
            import traceback
            traceback.print_exc()


def main():
    """Entry point for the GUI process."""
    log(LogTag.GUI, "Starting...")

    try:
        processor = GUIProcessor()
        processor.run()
    except Exception as e:
        log(LogTag.GUI, f"Fatal error — {e}", level="ERR")
        import traceback
        traceback.print_exc()
    finally:
        log(LogTag.GUI, "Shutting down...")
        cleanup_manager()


if __name__ == "__main__":
    main()