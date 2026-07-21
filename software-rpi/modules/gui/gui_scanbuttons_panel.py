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
Scanning control buttons panel

Drives the film transport and the scanning sequence. Shows the frame counter and
provides start/stop controls, manual positioning by marker, frame, step or ten
frames, and the hardware toggles for the serial link and the stepper motors.
Button availability follows a three-tier hierarchy: the serial toggle is always
live, the stepper toggle unlocks once serial connects, and the movement buttons
unlock once the steppers are enabled.

Shared memory:
    Reads:  STEPPERS_ENABLED, SERIAL_CONNECTED, STEPPERS_ENABLE_FAILED,
            SERIAL_CONNECT_FAILED, FRAME_COUNTER (uint16)
    Writes: CMD_SEQ_START, CMD_SEQ_STOP, CMD_MOVE_TO_MARKER, CMD_FRAME_FWD,
            CMD_FRAME_REV, CMD_STEP_FWD, CMD_STEP_REV, CMD_X_FRAMES_FWD,
            CMD_X_FRAMES_REV, CMD_ENABLE_STEPPERS, CMD_DISABLE_STEPPERS,
            SERIAL_CONNECT_REQUEST, SERIAL_DISCONNECT_REQUEST,
            BUTTON_PRESSED, BUTTON_DONE, FRAME_COUNTER (uint16),
            STEPPERS_ENABLE_FAILED and SERIAL_CONNECT_FAILED (cleared once shown)

Interacts with:
    mod_controller (CMD_* requests, BUTTON_PRESSED / BUTTON_DONE handshake,
                    STEPPERS_ENABLED, SERIAL_CONNECTED, frame counter)
    mod_serial     (connect and disconnect requests, failure reporting)
    mod_gui        (instantiated as a panel; this panel also runs its own timers)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib, Pango

from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex, Uint16Index
from tpx_logger import log, LogTag

class ScanButtonsPanel(BasePanel):
    """Panel for the scanning sequence, manual film positioning, and hardware toggles."""

    def __init__(self, flags, shared_arrays):
        """Bind the shared counter array and set up the state tracking used to detect button-state changes."""
        super().__init__("Scanning Control", flags, shared_arrays)

        self.sm_uint16var = shared_arrays['sm_uint16var']

        # Track displayed counter value to detect changes
        self.displayed_counter = 0
        self.counter_label = None

        # Transport control buttons (will be populated in create_frame)
        self.btn_start = None
        self.btn_stop = None
        self.btn_to_marker = None
        self.btn_frame_fwd = None
        self.btn_frame_rev = None
        self.btn_step_fwd = None
        self.btn_step_rev = None
        self.btn_continuous_rev = None
        self.btn_continuous_fwd = None

        # Transport control toggle buttons
        self.btn_stepper_toggle = None
        self.btn_serial_toggle = None

        # Track previous states to detect changes
        self.prev_steppers_enabled = False
        self.prev_serial_connected = False
        self.prev_controls_locked = False  # Start opposite to force initial update

        log(LogTag.GUI, "Scan Buttons Panel - Initialized")

    def create_frame(self):
        """Build the panel: frame counter and sequence controls on top, film move and hardware frames below."""
        frame, vbox = self._create_base_frame()
        frame.set_halign(Gtk.Align.START)

        # Top half - Sequence Controls

        # Frame counter display
        counter_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        counter_box.set_halign(Gtk.Align.CENTER)

        counter_label_text = Gtk.Label(label="Frame:")
        attrs_title = Pango.AttrList()
        attrs_title.insert(Pango.attr_scale_new(2.0))
        attrs_title.insert(Pango.attr_weight_new(Pango.Weight.BOLD))
        counter_label_text.set_attributes(attrs_title)
        counter_box.append(counter_label_text)

        self.counter_label = Gtk.Label(label="000000")
        self.counter_label.set_width_chars(6)
        self.counter_label.set_xalign(1.0)  # Right-align the number
        attrs = Pango.AttrList()
        attrs.insert(Pango.attr_scale_new(2.0))
        attrs.insert(Pango.attr_weight_new(Pango.Weight.BOLD))
        self.counter_label.set_attributes(attrs)
        counter_box.append(self.counter_label)

        vbox.append(counter_box)

        # Reset Counter button
        btn_reset = Gtk.Button(label="Reset Counter")
        btn_reset.connect("clicked", self._on_reset_counter)
        vbox.append(btn_reset)

        # Start Sequence button
        self.btn_start = Gtk.Button(label="Start Sequence")
        self.btn_start.connect("clicked", self._on_start_sequence)
        vbox.append(self.btn_start)

        # Stop button
        self.btn_stop = Gtk.Button(label="Stop")
        self.btn_stop.connect("clicked", self._on_stop)
        vbox.append(self.btn_stop)

        # Middle - Film Move Frame
        filmmove_frame = Gtk.Frame(label="Film Move")
        filmmove_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        filmmove_vbox.set_margin_top(6)
        filmmove_vbox.set_margin_bottom(6)
        filmmove_vbox.set_margin_start(6)
        filmmove_vbox.set_margin_end(6)

        # To Marker button
        self.btn_to_marker = Gtk.Button(label="To Marker")
        self.btn_to_marker.connect("clicked", self._on_to_marker)
        filmmove_vbox.append(self.btn_to_marker)

        # Frame REV / FWD row
        frame_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.btn_frame_rev = Gtk.Button(label="Frame REV")
        self.btn_frame_rev.connect("clicked", self._on_frame_rev)
        self.btn_frame_rev.set_hexpand(True)
        frame_hbox.append(self.btn_frame_rev)
        self.btn_frame_fwd = Gtk.Button(label="Frame FWD")
        self.btn_frame_fwd.connect("clicked", self._on_frame_fwd)
        self.btn_frame_fwd.set_hexpand(True)
        frame_hbox.append(self.btn_frame_fwd)
        filmmove_vbox.append(frame_hbox)

        # Step REV / FWD row
        step_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.btn_step_rev = Gtk.Button(label="Step REV")
        self.btn_step_rev.connect("clicked", self._on_step_rev)
        self.btn_step_rev.set_hexpand(True)
        step_hbox.append(self.btn_step_rev)
        self.btn_step_fwd = Gtk.Button(label="Step FWD")
        self.btn_step_fwd.connect("clicked", self._on_step_fwd)
        self.btn_step_fwd.set_hexpand(True)
        step_hbox.append(self.btn_step_fwd)
        filmmove_vbox.append(step_hbox)

        # 10 Frames REV / FWD row
        xframes_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.btn_continuous_rev = Gtk.Button(label="10 Frames REV")
        self.btn_continuous_rev.connect("clicked", self._on_continuous_rev)
        self.btn_continuous_rev.set_hexpand(True)
        xframes_hbox.append(self.btn_continuous_rev)
        self.btn_continuous_fwd = Gtk.Button(label="10 Frames FWD")
        self.btn_continuous_fwd.connect("clicked", self._on_continuous_fwd)
        self.btn_continuous_fwd.set_hexpand(True)
        xframes_hbox.append(self.btn_continuous_fwd)
        filmmove_vbox.append(xframes_hbox)

        filmmove_frame.set_child(filmmove_vbox)
        vbox.append(filmmove_frame)

        # Bottom - Hardware Control Frame
        hardware_frame = Gtk.Frame(label="Hardware")
        hardware_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        hardware_vbox.set_margin_top(6)
        hardware_vbox.set_margin_bottom(6)
        hardware_vbox.set_margin_start(6)
        hardware_vbox.set_margin_end(6)

        self.btn_stepper_toggle = Gtk.Button(label="Steppers — Enable All")
        self.btn_stepper_toggle.connect("clicked", self._on_stepper_toggle)
        hardware_vbox.append(self.btn_stepper_toggle)

        self.btn_serial_toggle = Gtk.Button(label="Serial Connect")
        self.btn_serial_toggle.connect("clicked", self._on_serial_toggle)
        hardware_vbox.append(self.btn_serial_toggle)

        hardware_frame.set_child(hardware_vbox)
        vbox.append(hardware_frame)

        # Force initial button states before returning
        # FIRST TIER: Always available (not touched here - Reset Counter, Serial Connect button)
        # SECOND TIER: Stepper button - locked until serial connected
        self.btn_stepper_toggle.set_sensitive(False)
        # THIRD TIER: Movement buttons - locked until steppers enabled
        self.btn_start.set_sensitive(False)
        self.btn_stop.set_sensitive(False)
        self.btn_to_marker.set_sensitive(False)
        self.btn_frame_fwd.set_sensitive(False)
        self.btn_frame_rev.set_sensitive(False)
        self.btn_step_fwd.set_sensitive(False)
        self.btn_step_rev.set_sensitive(False)
        self.btn_continuous_rev.set_sensitive(False)
        self.btn_continuous_fwd.set_sensitive(False)

        # Start periodic counter update (100ms)
        GLib.timeout_add(100, self._update_counter)

        # Start periodic transport control state update (100ms)
        GLib.timeout_add(100, self._update_transport_controls)

        return frame

    # Sequence Control Handlers

    def _on_reset_counter(self, button):
        """Zero the frame counter."""
        self.sm_uint16var[Uint16Index.FRAME_COUNTER] = 0
        log(LogTag.GUI, "Reset Counter clicked")

    def _on_start_sequence(self, button):
        """Request the start of an automatic scanning sequence."""
        self.flags[FlagIndex.CMD_SEQ_START] = True # Start Scanning Sequence
        log(LogTag.GUI, "Start Sequence clicked")

    def _on_stop(self, button):
        """Request that the scanning sequence stop."""
        self.flags[FlagIndex.CMD_SEQ_STOP] = True # Stop Scanning Sequence
        log(LogTag.GUI, "Stop clicked")

    # Manual Positioning Handlers

    def _on_to_marker(self, button):
        """Request a move to the sprocket marker."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_MOVE_TO_MARKER] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True

        log(LogTag.GUI, "To Marker clicked")

    def _on_frame_fwd(self, button):
        """Request a one-frame move forward."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_FRAME_FWD] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "Frame Fwd clicked")

    def _on_frame_rev(self, button):
        """Request a one-frame move in reverse."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_FRAME_REV] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "Frame Rev clicked")

    def _on_step_fwd(self, button):
        """Request a single stepper step forward."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_STEP_FWD] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "Step Fwd clicked")

    def _on_step_rev(self, button):
        """Request a single stepper step in reverse."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_STEP_REV] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "Step Rev clicked")

    def _on_continuous_rev(self, button):
        """Request a ten-frame move in reverse, roughly a second of travel."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_X_FRAMES_REV] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "10 Frames REV clicked")

    def _on_continuous_fwd(self, button):
        """Request a ten-frame move forward, roughly a second of travel."""
        self.flags[FlagIndex.BUTTON_DONE] = False
        self.flags[FlagIndex.CMD_X_FRAMES_FWD] = True
        self.flags[FlagIndex.BUTTON_PRESSED] = True
        log(LogTag.GUI, "10 Frames FWD clicked")

    # Transport Control Handlers

    def _on_stepper_toggle(self, button):
        """Request that the steppers be enabled or disabled, depending on their current state."""
        steppers_enabled = self.flags[FlagIndex.STEPPERS_ENABLED]

        if steppers_enabled:
            # Currently enabled, request disable
            self.flags[FlagIndex.CMD_DISABLE_STEPPERS] = True
            log(LogTag.GUI, "Disable Steppers clicked")
        else:
            # Currently disabled, request enable
            self.flags[FlagIndex.CMD_ENABLE_STEPPERS] = True
            log(LogTag.GUI, "Enable Steppers clicked")

    def _on_serial_toggle(self, button):
        """Request that the serial link connect or disconnect, depending on its current state."""
        serial_connected = self.flags[FlagIndex.SERIAL_CONNECTED]

        if serial_connected:
            # Currently connected, request disconnect
            self.flags[FlagIndex.SERIAL_DISCONNECT_REQUEST] = True
            log(LogTag.GUI, "Disconnect Serial clicked")
        else:
            # Currently disconnected, request connect
            self.flags[FlagIndex.SERIAL_CONNECT_REQUEST] = True
            log(LogTag.GUI, "Connect Serial clicked")

    def _update_transport_controls(self):
        """Refresh button sensitivity, labels and error styling from the current hardware state, and keep the timer running."""
        try:
            # Read current states from flags
            steppers_enabled = self.flags[FlagIndex.STEPPERS_ENABLED]
            serial_connected = self.flags[FlagIndex.SERIAL_CONNECTED]
            stepper_enable_failed = self.flags[FlagIndex.STEPPERS_ENABLE_FAILED]
            serial_connect_failed = self.flags[FlagIndex.SERIAL_CONNECT_FAILED]

            # Compute lock state: lock if steppers disabled OR serial disconnected
            controls_locked = (not steppers_enabled) or (not serial_connected)

            # Update transport button sensitivity based on three-tier hierarchy
            # Check BEFORE updating prev_ values
            condition1 = controls_locked != self.prev_controls_locked
            condition2 = serial_connected != self.prev_serial_connected
            condition3 = steppers_enabled != self.prev_steppers_enabled

            if condition1 or condition2 or condition3:
                # FIRST TIER: Always available (Reset Counter, Serial toggle)
                # Serial toggle is always available - no change needed
                # Reset Counter is not in this panel's button list - always available

                # SECOND TIER: Stepper toggle - available when serial connected, locked when disconnected
                self.btn_stepper_toggle.set_sensitive(serial_connected)

                # THIRD TIER: Movement buttons - available only when steppers enabled (which requires serial connected)
                # controls_locked = True when steppers disabled OR serial disconnected
                self.btn_start.set_sensitive(not controls_locked)
                self.btn_stop.set_sensitive(not controls_locked)
                self.btn_to_marker.set_sensitive(not controls_locked)
                self.btn_frame_fwd.set_sensitive(not controls_locked)
                self.btn_frame_rev.set_sensitive(not controls_locked)
                self.btn_step_fwd.set_sensitive(not controls_locked)
                self.btn_step_rev.set_sensitive(not controls_locked)
                self.btn_continuous_rev.set_sensitive(not controls_locked)
                self.btn_continuous_fwd.set_sensitive(not controls_locked)

                self.prev_controls_locked = controls_locked

            # Update stepper toggle button text if state changed
            if steppers_enabled != self.prev_steppers_enabled:
                if steppers_enabled:
                    self.btn_stepper_toggle.set_label("Steppers — Disable All")
                else:
                    self.btn_stepper_toggle.set_label("Steppers — Enable All")
                self.prev_steppers_enabled = steppers_enabled

            # Update serial toggle button text if state changed
            if serial_connected != self.prev_serial_connected:
                if serial_connected:
                    self.btn_serial_toggle.set_label("Serial Disconnect")
                else:
                    self.btn_serial_toggle.set_label("Serial Connect")
                self.prev_serial_connected = serial_connected

            # Update button backgrounds for error states
            if stepper_enable_failed:
                self._set_button_error_style(self.btn_stepper_toggle, True)
                self.flags[FlagIndex.STEPPERS_ENABLE_FAILED] = False  # Clear after applying
            else:
                self._set_button_error_style(self.btn_stepper_toggle, False)

            if serial_connect_failed:
                self._set_button_error_style(self.btn_serial_toggle, True)
                self.flags[FlagIndex.SERIAL_CONNECT_FAILED] = False  # Clear after applying
            else:
                self._set_button_error_style(self.btn_serial_toggle, False)

            return True  # Continue periodic updates

        except Exception as e:
            log(LogTag.GUI, f"Transport controls update error — {e}", level="ERR")
            return True

    def _set_button_error_style(self, button, is_error):
        """Apply or remove the red error background on a button."""
        if is_error:
            # Apply dark red background using CSS
            css_provider = Gtk.CssProvider()
            css_provider.load_from_data(b"""
                button {
                    background-color: #8B0000;
                    color: #FFFFFF;
                }
            """)
            button.get_style_context().add_provider(
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            # Store provider reference so we can remove it later
            button._error_css_provider = css_provider
        else:
            # Remove custom styling if it exists
            if hasattr(button, '_error_css_provider'):
                button.get_style_context().remove_provider(button._error_css_provider)
                delattr(button, '_error_css_provider')

    # Counter Update

    def _update_counter(self):
        """Refresh the frame counter readout when it changes, and keep the timer running."""
        try:
            current_counter = self.sm_uint16var[Uint16Index.FRAME_COUNTER]

            if current_counter != self.displayed_counter:
                self.displayed_counter = current_counter
                self.counter_label.set_text(f"{current_counter:06d}")

            return True  # Continue periodic updates

        except Exception as e:
            log(LogTag.GUI, f"Scan Buttons Panel counter update error — {e}", level="ERR")
            return True

    def update_display(self):
        """Nothing to refresh from the main loop; this panel runs its own timers."""
        # Counter updates via periodic timer, not the main display loop
        return True