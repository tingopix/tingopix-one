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
Exposure control panel

Provides the exposure and illumination controls for scanning. A pair of live
sliders set the working exposure and LED level, which can be pushed straight to
the camera and light source or stored into one of the per-channel blocks
(Detection, ALL, or separate R/G/B). Also selects the capture mode and stack
depth, triggers single captures, and saves or loads exposure presets.

Shared memory:
    Reads:  EXP_RGB, EXP_DET, EXP_R, EXP_G, EXP_B,
            LED_RGB, LED_DET, LED_R, LED_G, LED_B, STACKING_SETTING
    Writes: MODE_CAPTURE_RGB, RQST_CAPTURE, READY_CAPTURE, RQST_LED_TAKE,
            STACKING_SETTING, EXPOSURE, LED_TAKE,
            EXP_RGB, EXP_DET, EXP_R, EXP_G, EXP_B,
            LED_RGB, LED_DET, LED_R, LED_G, LED_B

Interacts with:
    mod_capture      (RQST_CAPTURE / READY_CAPTURE handshake, EXPOSURE, EXP_*)
    mod_serial       (RQST_LED_TAKE, LED_TAKE, LED_* levels)
    mod_controller   (MODE_CAPTURE_RGB, STACKING_SETTING)
    mod_exp_defaults (preset save and load)
    mod_gui          (instantiated as a panel, refreshed on the GUI timer)
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gio


from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex, Uint16Index, Uint32Index
from modules.mod_exp_defaults import save_preset, load_preset, PRESETS_DIR, ensure_presets_dir
from tpx_logger import log, LogTag


class ScanControlPanel(BasePanel):
    """Panel for exposure and light control, capture mode, stacking, and presets."""

    def __init__(self, flags, shared_arrays):
        """Bind the shared arrays and mirror the stored exposure and LED settings into local state."""
        super().__init__("Exposure Control", flags, shared_arrays)

        self.uint16_var = shared_arrays['sm_uint16var']
        self.uint32_var = shared_arrays['sm_uint32var']
        self.float_var  = shared_arrays['sm_floatvar']

        self.uint16_var[Uint16Index.STACKING_SETTING] = 1 #Initialize stacking to startup value.

        # Read exposure and light values from shared memory (already initialized by mod_capture)
        # Do NOT call initialize_defaults() here - it would overwrite loaded presets!

        # Initialize GUI-specific flags (not in defaults module)
        self.flags[FlagIndex.MODE_CAPTURE_RGB] = False

        # Read back from shared memory to class variables (single source of truth)
        self.rgb_exposure = self.uint32_var[Uint32Index.EXP_RGB]
        self.rgb_light = self.uint16_var[Uint16Index.LED_RGB]
        self.det_exposure = self.uint32_var[Uint32Index.EXP_DET]
        self.det_light = self.uint16_var[Uint16Index.LED_DET]
        self.r_exposure = self.uint32_var[Uint32Index.EXP_R]
        self.r_light = self.uint16_var[Uint16Index.LED_R]
        self.g_exposure = self.uint32_var[Uint32Index.EXP_G]
        self.g_light = self.uint16_var[Uint16Index.LED_G]
        self.b_exposure = self.uint32_var[Uint32Index.EXP_B]
        self.b_light = self.uint16_var[Uint16Index.LED_B]

        # Mode radio buttons
        self.mode_all_radio = None
        self.mode_rgb_radio = None

        # Capture button
        self.capture_button = None

        # Master controls
        self.stacking_combo = None
        self.light_slider = None
        self.light_value_label = None
        self.light_take_button = None
        self.exposure_slider = None
        self.exposure_value_label = None
        self.exposure_take_button = None

        # Detection block
        self.det_exp_label = None
        self.det_light_label = None

        # ALL block
        self.all_exp_label = None
        self.all_light_label = None
        self.all_block = None

        # RGB channel blocks container
        self.rgb_channels_box = None

        # Channel value labels
        self.r_exp_label = None
        self.r_light_label = None
        self.g_exp_label = None
        self.g_light_label = None
        self.b_exp_label = None
        self.b_light_label = None

        # Preset buttons
        self.btn_save_preset = None
        self.btn_load_preset = None

        log(LogTag.GUI, "Scan Control Panel - Initialized")

    def create_frame(self):
        """Build the panel: exposure and light sliders on top, then the channel blocks, settings and capture controls."""
        frame, vbox = self._create_base_frame()

        # Exposure slider — full width
        exp_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        exp_hbox.set_margin_bottom(4)

        exp_label = Gtk.Label(label="Exposure:")
        exp_label.set_xalign(0)
        exp_label.set_size_request(60, -1)
        exp_hbox.append(exp_label)

        self.exposure_slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 2, 80, 1
        )
        self.exposure_slider.set_value(self.rgb_exposure // 1200)
        self.exposure_slider.set_draw_value(False)
        self.exposure_slider.set_hexpand(True)
        self.exposure_slider.connect("value-changed", self._on_exposure_changed)
        exp_hbox.append(self.exposure_slider)

        self.exposure_value_label = Gtk.Label(label=f"{self.rgb_exposure} µs")
        self.exposure_value_label.set_size_request(72, -1)
        exp_hbox.append(self.exposure_value_label)

        self.exposure_take_button = Gtk.Button(label="Take EXP")
        self.exposure_take_button.connect("clicked", self._on_exposure_take_clicked)
        exp_hbox.append(self.exposure_take_button)

        vbox.append(exp_hbox)

        # Light slider — full width
        light_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        light_hbox.set_margin_bottom(8)

        light_label = Gtk.Label(label="Light:")
        light_label.set_xalign(0)
        light_label.set_size_request(60, -1)
        light_hbox.append(light_label)

        self.light_slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0, 4095, 1
        )
        self.rgb_light = self.uint16_var[Uint16Index.LED_RGB]
        self.light_slider.set_value(self.rgb_light)
        self.light_slider.set_draw_value(False)
        self.light_slider.set_hexpand(True)
        self.light_slider.connect("value-changed", self._on_light_changed)
        light_hbox.append(self.light_slider)

        self.light_value_label = Gtk.Label(label=f"{self.rgb_light:04d}")
        self.light_value_label.set_size_request(72, -1)
        light_hbox.append(self.light_value_label)

        self.light_take_button = Gtk.Button(label="Take LED")
        self.light_take_button.connect("clicked", self._on_light_take_clicked)
        light_hbox.append(self.light_take_button)

        vbox.append(light_hbox)

        # Main horizontal row: Capture Control | Detection | ALL or R G B
        main_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        # Detection block
        det_frame = self._create_channel_block(
            "Detection",
            self._on_det_set_exposure, self._on_det_set_light
        )
        main_hbox.append(det_frame)

        # ALL block (All mode)
        self.all_block = self._create_channel_block(
            "ALL",
            self._on_all_set_exposure, self._on_all_set_light
        )
        self.all_block.set_visible(True)
        main_hbox.append(self.all_block)

        # RGB channel blocks (RGB mode)
        self.rgb_channels_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        for ch_name, exp_cb, light_cb in [
            ("R", self._on_r_set_exposure, self._on_r_set_light),
            ("G", self._on_g_set_exposure, self._on_g_set_light),
            ("B", self._on_b_set_exposure, self._on_b_set_light),
        ]:
            self.rgb_channels_box.append(
                self._create_channel_block(ch_name, exp_cb, light_cb)
            )

        self.rgb_channels_box.set_visible(False)
        main_hbox.append(self.rgb_channels_box)

        # Settings frame
        settings_frame = Gtk.Frame(label="Settings")
        settings_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        settings_vbox.set_margin_top(6)
        settings_vbox.set_margin_bottom(6)
        settings_vbox.set_margin_start(6)
        settings_vbox.set_margin_end(6)

        self.btn_load_preset = Gtk.Button(label="Load")
        self.btn_load_preset.connect("clicked", self._on_load_preset_clicked)
        self.btn_load_preset.set_hexpand(True)
        settings_vbox.append(self.btn_load_preset)

        self.btn_save_preset = Gtk.Button(label="Save")
        self.btn_save_preset.connect("clicked", self._on_save_preset_clicked)
        self.btn_save_preset.set_hexpand(True)
        settings_vbox.append(self.btn_save_preset)

        settings_frame.set_child(settings_vbox)
        main_hbox.append(settings_frame)

        # Capture Control frame (rightmost)
        capture_frame = Gtk.Frame(label="Capture Control")
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        left_vbox.set_margin_top(6)
        left_vbox.set_margin_bottom(6)
        left_vbox.set_margin_start(6)
        left_vbox.set_margin_end(6)

        mode_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        mode_label = Gtk.Label(label="Mode:")
        mode_hbox.append(mode_label)

        self.mode_all_radio = Gtk.CheckButton(label="All")
        self.mode_all_radio.set_active(True)
        self.mode_all_radio.connect("toggled", self._on_mode_changed)
        mode_hbox.append(self.mode_all_radio)

        self.mode_rgb_radio = Gtk.CheckButton(label="RGB")
        self.mode_rgb_radio.set_group(self.mode_all_radio)
        self.mode_rgb_radio.connect("toggled", self._on_mode_changed)
        mode_hbox.append(self.mode_rgb_radio)

        left_vbox.append(mode_hbox)

        stacking_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        stacking_label = Gtk.Label(label="Stacking:")
        stacking_label.set_xalign(0)
        stacking_hbox.append(stacking_label)

        self.stacking_combo = Gtk.ComboBoxText()
        for val in ["1", "2", "4", "8", "16"]:
            self.stacking_combo.append_text(val)
        current_stacking = self.uint16_var[Uint16Index.STACKING_SETTING]
        stacking_index = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}.get(current_stacking, 0)
        self.stacking_combo.set_active(stacking_index)
        self.stacking_combo.connect("changed", self._on_stacking_changed)
        self.stacking_combo.set_hexpand(True)
        stacking_hbox.append(self.stacking_combo)
        left_vbox.append(stacking_hbox)

        self.capture_button = Gtk.Button(label="Capture")
        self.capture_button.connect("clicked", self._on_capture_clicked)
        self.capture_button.set_hexpand(True)
        left_vbox.append(self.capture_button)

        capture_frame.set_child(left_vbox)
        main_hbox.append(capture_frame)

        vbox.append(main_hbox)

        return frame

    def _create_channel_block(self, channel_name, exp_callback, light_callback):
        """Build one channel block showing its stored exposure and light values with buttons to overwrite them.

        Returns the frame, and remembers the two value labels so later updates can refresh them.
        """
        if channel_name == "Detection":
            exp_val = f"{self.det_exposure} µs"
            light_val = f"{self.det_light:04d}"
        elif channel_name == "ALL":
            exp_val = f"{self.rgb_exposure} µs"
            light_val = f"{self.rgb_light:04d}"
        elif channel_name == "R":
            exp_val = f"{self.r_exposure} µs"
            light_val = f"{self.r_light:04d}"
        elif channel_name == "G":
            exp_val = f"{self.g_exposure} µs"
            light_val = f"{self.g_light:04d}"
        elif channel_name == "B":
            exp_val = f"{self.b_exposure} µs"
            light_val = f"{self.b_light:04d}"
        else:
            exp_val = "0 µs"
            light_val = "000"

        frame = Gtk.Frame(label=channel_name)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        vbox.set_margin_top(4)
        vbox.set_margin_bottom(4)
        vbox.set_margin_start(4)
        vbox.set_margin_end(4)

        exp_label = Gtk.Label(label=exp_val)
        vbox.append(exp_label)

        set_exp_btn = Gtk.Button(label="Set EXP")
        set_exp_btn.connect("clicked", exp_callback)
        vbox.append(set_exp_btn)

        light_label = Gtk.Label(label=light_val)
        vbox.append(light_label)

        set_light_btn = Gtk.Button(label="Set LED")
        set_light_btn.connect("clicked", light_callback)
        vbox.append(set_light_btn)

        frame.set_child(vbox)

        # Store label references
        if channel_name == "Detection":
            self.det_exp_label = exp_label
            self.det_light_label = light_label
        elif channel_name == "ALL":
            self.all_exp_label = exp_label
            self.all_light_label = light_label
        elif channel_name == "R":
            self.r_exp_label = exp_label
            self.r_light_label = light_label
        elif channel_name == "G":
            self.g_exp_label = exp_label
            self.g_light_label = light_label
        elif channel_name == "B":
            self.b_exp_label = exp_label
            self.b_light_label = light_label

        return frame

    def _on_mode_changed(self, radio_button):
        """Switch between the ALL block and the separate R/G/B blocks, and publish the capture mode."""
        if self.mode_rgb_radio.get_active():
            self.all_block.set_visible(False)
            self.rgb_channels_box.set_visible(True)
            self.flags[FlagIndex.MODE_CAPTURE_RGB] = True
            log(LogTag.GUI, "Scan mode changed to RGB")
        else:
            self.rgb_channels_box.set_visible(False)
            self.all_block.set_visible(True)
            self.flags[FlagIndex.MODE_CAPTURE_RGB] = False
            log(LogTag.GUI, "Scan mode changed to All Channels")

    def _on_capture_clicked(self, button):
        """Request a single capture."""
        log(LogTag.GUI, "Capture button clicked")
        self.flags[FlagIndex.RQST_CAPTURE] = True
        self.flags[FlagIndex.READY_CAPTURE] = False

    def _on_stacking_changed(self, combo):
        """Publish the selected stack depth."""
        text = combo.get_active_text()
        if text:
            value = int(text)
            self.uint16_var[Uint16Index.STACKING_SETTING] = value
            log(LogTag.GUI, f"Stacking value changed to {value}")

    def _on_light_changed(self, scale):
        """Update the light readout as the slider moves, without touching the hardware."""
        value = int(scale.get_value())
        self.light_value_label.set_text(f"{value:04d}")

    def _on_light_take_clicked(self, button):
        """Push the slider's light value to the live LED without storing it as a channel setting."""
        value = int(self.light_slider.get_value())
        self.uint16_var[Uint16Index.LED_TAKE] = int(value)
        self.flags[FlagIndex.RQST_LED_TAKE] = True
        log(LogTag.GUI, f"Light value applied: 0x{value:03X} ({value})")

    def _on_exposure_changed(self, scale):
        """Update the exposure readout as the slider moves, without touching the camera."""
        value = int(scale.get_value())
        exposure_us = value * 1200
        self.exposure_value_label.set_text(f"{exposure_us} µs")

    def _on_exposure_take_clicked(self, button):
        """Push the slider's exposure to the live camera without storing it as a channel setting."""
        value = int(self.exposure_slider.get_value())
        exposure_us = value * 1200
        self.uint32_var[Uint32Index.EXPOSURE] = exposure_us
        log(LogTag.GUI, f"Exposure applied: {value} → {exposure_us} µs")

    def _get_current_exposure(self):
        """Return the slider's exposure in microseconds."""
        return int(self.exposure_slider.get_value()) * 1200

    def _get_current_light(self):
        """Return the slider's light value."""
        return int(self.light_slider.get_value())

    def _on_all_set_exposure(self, button):
        """Store the slider's exposure as the ALL-channel setting."""
        self.rgb_exposure = self._get_current_exposure()
        self.uint32_var[Uint32Index.EXP_RGB] = self.rgb_exposure
        self.all_exp_label.set_text(f"{self.rgb_exposure} µs")
        log(LogTag.GUI, f"ALL exposure set to {self.rgb_exposure} µs")

    def _on_all_set_light(self, button):
        """Store the slider's light as the ALL-channel setting. This is a stored value only; it does not drive the LED."""
        self.rgb_light = self._get_current_light()
        self.uint16_var[Uint16Index.LED_RGB] = self.rgb_light
        self.all_light_label.set_text(f"{self.rgb_light:04d}")
        log(LogTag.GUI, f"ALL light set to 0x{self.rgb_light:03X}")

    def _on_det_set_exposure(self, button):
        """Store the slider's exposure as the Detection setting."""
        self.det_exposure = self._get_current_exposure()
        self.uint32_var[Uint32Index.EXP_DET] = self.det_exposure
        self.det_exp_label.set_text(f"{self.det_exposure} µs")
        log(LogTag.GUI, f"Detection exposure set to {self.det_exposure} µs")

    def _on_det_set_light(self, button):
        """Store the slider's light as the Detection setting."""
        self.det_light = self._get_current_light()
        self.uint16_var[Uint16Index.LED_DET] = self.det_light
        self.det_light_label.set_text(f"{self.det_light:04d}")
        log(LogTag.GUI, f"Detection light set to 0x{self.det_light:03X}")

    def _on_r_set_exposure(self, button):
        """Store the slider's exposure as the Red-channel setting."""
        self.r_exposure = self._get_current_exposure()
        self.uint32_var[Uint32Index.EXP_R] = self.r_exposure
        self.r_exp_label.set_text(f"{self.r_exposure} µs")
        log(LogTag.GUI, f"Red exposure set to {self.r_exposure} µs")

    def _on_r_set_light(self, button):
        """Store the slider's light as the Red-channel setting."""
        self.r_light = self._get_current_light()
        self.uint16_var[Uint16Index.LED_R] = self.r_light
        self.r_light_label.set_text(f"{self.r_light:04d}")
        log(LogTag.GUI, f"Red light set to 0x{self.r_light:03X}")

    def _on_g_set_exposure(self, button):
        """Store the slider's exposure as the Green-channel setting."""
        self.g_exposure = self._get_current_exposure()
        self.uint32_var[Uint32Index.EXP_G] = self.g_exposure
        self.g_exp_label.set_text(f"{self.g_exposure} µs")
        log(LogTag.GUI, f"Green exposure set to {self.g_exposure} µs")

    def _on_g_set_light(self, button):
        """Store the slider's light as the Green-channel setting."""
        self.g_light = self._get_current_light()
        self.uint16_var[Uint16Index.LED_G] = self.g_light
        self.g_light_label.set_text(f"{self.g_light:04d}")
        log(LogTag.GUI, f"Green light set to 0x{self.g_light:03X}")

    def _on_b_set_exposure(self, button):
        """Store the slider's exposure as the Blue-channel setting."""
        self.b_exposure = self._get_current_exposure()
        self.uint32_var[Uint32Index.EXP_B] = self.b_exposure
        self.b_exp_label.set_text(f"{self.b_exposure} µs")
        log(LogTag.GUI, f"Blue exposure set to {self.b_exposure} µs")

    def _on_b_set_light(self, button):
        """Store the slider's light as the Blue-channel setting."""
        self.b_light = self._get_current_light()
        self.uint16_var[Uint16Index.LED_B] = self.b_light
        self.b_light_label.set_text(f"{self.b_light:04d}")
        log(LogTag.GUI, f"Blue light set to 0x{self.b_light:03X}")

    def get_mode(self):
        """Return the selected capture mode, either RGB or All."""
        return "RGB" if self.mode_rgb_radio.get_active() else "All"

    def update_display(self):
        """Nothing to refresh; this panel updates only in response to user input."""
        return True

    def _on_save_preset_clicked(self, button):
        """Open the preset save dialog."""
        dialog = Gtk.FileChooserDialog(
            title="Save Exposure Preset",
            action=Gtk.FileChooserAction.SAVE,
        )
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Save", Gtk.ResponseType.ACCEPT)
        ensure_presets_dir()
        dialog.set_current_folder(Gio.File.new_for_path(str(PRESETS_DIR)))
        dialog.set_current_name("exposure_defaults.json")
        filter_json = Gtk.FileFilter()
        filter_json.set_name("TingoPix Presets (*.json)")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)
        dialog.set_modal(True)
        dialog.set_transient_for(button.get_root())
        dialog.connect("response", self._on_save_preset_response)
        dialog.show()

    def _on_save_preset_response(self, dialog, response):
        """Write the current exposure and light settings to the chosen preset file."""
        if response == Gtk.ResponseType.ACCEPT:
            file = dialog.get_file()
            if file:
                filepath = file.get_path()
                if not filepath.endswith('.json'):
                    filepath += '.json'
                success = save_preset(self.uint32_var, self.uint16_var, filepath, self.float_var, mode=self.get_mode())
                if success:
                    log(LogTag.GUI, f"Preset saved to {filepath}")
                else:
                    log(LogTag.GUI, "Failed to save preset", level="WARN")
        dialog.destroy()

    def _on_load_preset_clicked(self, button):
        """Open the preset load dialog."""
        dialog = Gtk.FileChooserDialog(
            title="Load Exposure Preset",
            action=Gtk.FileChooserAction.OPEN,
        )
        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Load", Gtk.ResponseType.ACCEPT)
        ensure_presets_dir()
        dialog.set_current_folder(Gio.File.new_for_path(str(PRESETS_DIR)))
        filter_json = Gtk.FileFilter()
        filter_json.set_name("TingoPix Presets (*.json)")
        filter_json.add_pattern("*.json")
        dialog.add_filter(filter_json)
        dialog.set_modal(True)
        dialog.set_transient_for(button.get_root())
        dialog.connect("response", self._on_load_preset_response)
        dialog.show()

    def _on_load_preset_response(self, dialog, response):
        """Load the chosen preset into shared memory and refresh the panel from it."""
        if response == Gtk.ResponseType.ACCEPT:
            file = dialog.get_file()
            if file:
                filepath = file.get_path()
                loaded_mode = load_preset(self.uint32_var, self.uint16_var, filepath, self.float_var)
                if loaded_mode is not None:
                    log(LogTag.GUI, f"Preset loaded from {filepath}")
                    self._update_gui_from_loaded_preset(loaded_mode)
                else:
                    log(LogTag.GUI, "Failed to load preset", level="WARN")
        dialog.destroy()

    def _update_gui_from_loaded_preset(self, mode="ALL"):
        """Refresh the channel labels and mode selection from the freshly loaded preset."""
        # Sliders are not updated — they represent transient live values,
        # not stored channel settings. User decides when to push via Take E / Take L.

        # Update class variables
        self.rgb_exposure = self.uint32_var[Uint32Index.EXP_RGB]
        self.rgb_light = self.uint16_var[Uint16Index.LED_RGB]
        self.det_exposure = self.uint32_var[Uint32Index.EXP_DET]
        self.det_light = self.uint16_var[Uint16Index.LED_DET]
        self.r_exposure = self.uint32_var[Uint32Index.EXP_R]
        self.r_light = self.uint16_var[Uint16Index.LED_R]
        self.g_exposure = self.uint32_var[Uint32Index.EXP_G]
        self.g_light = self.uint16_var[Uint16Index.LED_G]
        self.b_exposure = self.uint32_var[Uint32Index.EXP_B]
        self.b_light = self.uint16_var[Uint16Index.LED_B]

        # Update ALL block
        self.all_exp_label.set_text(f"{self.rgb_exposure} µs")
        self.all_light_label.set_text(f"{self.rgb_light:04d}")

        # Update Detection block
        self.det_exp_label.set_text(f"{self.det_exposure} µs")
        self.det_light_label.set_text(f"{self.det_light:04d}")

        # Update RGB channel blocks
        self.r_exp_label.set_text(f"{self.r_exposure} µs")
        self.r_light_label.set_text(f"{self.r_light:04d}")
        self.g_exp_label.set_text(f"{self.g_exposure} µs")
        self.g_light_label.set_text(f"{self.g_light:04d}")
        self.b_exp_label.set_text(f"{self.b_exposure} µs")
        self.b_light_label.set_text(f"{self.b_light:04d}")

        # Apply mode from preset
        if mode == "RGB":
            self.mode_rgb_radio.set_active(True)
        else:
            self.mode_all_radio.set_active(True)