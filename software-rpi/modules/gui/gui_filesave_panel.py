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
File saving control panel

Collects the destination and naming for saved frames and chooses the output
format. The save path, filename prefix, scene suffix and frame number are
written to shared memory as they are edited, so the file saver always reads
current values. Also selects sensor raw or debayered RGB output, toggles crop
and stabilisation, and issues save requests.

The string slots carry repurposed names: FILE_PATH holds the destination
directory, METADATA_1 the filename prefix, METADATA_2 the scene suffix, and
USER_COMMENT the frame number.

Shared memory:
    Reads:  SAVE_COMPLETE, USER_COMMENT
    Writes: SAVE_RAW, SAVE_APPLY_CROP, SAVE_REQUEST,
            SAVE_COMPLETE (cleared once the frame number has been refreshed),
            FILE_PATH, METADATA_1, METADATA_2, USER_COMMENT

Interacts with:
    mod_file_saver (save requests, naming strings, SAVE_COMPLETE handshake)
    mod_gui        (instantiated as a panel, refreshed on the GUI timer)
    tpx_config     (DEFAULT_FILEPATH, DEFAULT_PREFIX)
"""

import gi
from pathlib import Path
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from modules.gui.gui_base_panel import BasePanel
from modules.memory.shm_indexing import FlagIndex, StringIndex
from modules.memory.shm_manager import write_string, read_string
from tpx_config import DEFAULT_FILEPATH, DEFAULT_PREFIX
from tpx_logger import log, LogTag


class FileSavePanel(BasePanel):
    """Panel for choosing the save destination, naming, and output format."""

    def __init__(self, flags, shared_arrays):
        """Seed the default path, prefix, scene and frame number into shared memory and default to raw output."""
        super().__init__("File Saving", flags, shared_arrays)

        # Set default save path and prefix from tpx_config.
        # Expand "~" here so that both the entry box and shared memory carry an
        # absolute path — mod_file_saver creates the folder verbatim on save.
        self.default_path = str(Path(DEFAULT_FILEPATH).expanduser())

        # Entry widgets
        self.filepath_entry = None
        self.prefix_entry = None
        self.scene_entry = None
        self.frame_entry = None

        # Radio buttons for file type
        self.filetype_raw_radio = None
        self.filetype_debayered_radio = None

        # Crop checkbox
        self.crop_rgb_checkbox = None

        # Buttons
        self.browse_button = None
        self.save_button = None

        # Write default values to shared memory immediately
        write_string(StringIndex.FILE_PATH, self.default_path)
        write_string(StringIndex.METADATA_1, DEFAULT_PREFIX)  # default prefix
        write_string(StringIndex.METADATA_2, "001")        # default scene
        write_string(StringIndex.USER_COMMENT, "000001")   # default frame

        self.flags[FlagIndex.SAVE_RAW] = True  # Default to raw

        log(LogTag.GUI, "File Save Panel - Initialized")

    def create_frame(self):
        """Build the panel: file location fields on the left, output type and save button on the right."""
        frame, vbox = self._create_base_frame()

        # Top-level horizontal box: left (File Location) + right (File Type)
        columns_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        columns_hbox.set_margin_top(6)
        columns_hbox.set_margin_bottom(6)

        # Left column: File Location
        left_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        left_vbox.set_hexpand(True)

        left_header = Gtk.Label(label="File Location")
        left_header.set_xalign(0)
        left_vbox.append(left_header)

        left_grid = Gtk.Grid()
        left_grid.set_row_spacing(8)
        left_grid.set_column_spacing(12)

        # Filepath
        filepath_label = Gtk.Label(label="Filepath:")
        filepath_label.set_xalign(0)
        left_grid.attach(filepath_label, 0, 0, 1, 1)

        filepath_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        filepath_hbox.set_hexpand(True)
        self.filepath_entry = Gtk.Entry()
        self.filepath_entry.set_text(self.default_path)
        self.filepath_entry.set_hexpand(True)
        self.filepath_entry.set_width_chars(40)
        self.filepath_entry.connect("changed", self._on_filepath_changed)
        filepath_hbox.append(self.filepath_entry)

        self.browse_button = Gtk.Button(label="Browse...")
        self.browse_button.connect("clicked", self._on_browse_clicked)
        filepath_hbox.append(self.browse_button)

        left_grid.attach(filepath_hbox, 1, 0, 1, 1)

        # File Name Prefix
        prefix_label = Gtk.Label(label="File Name Prefix:")
        prefix_label.set_xalign(0)
        left_grid.attach(prefix_label, 0, 1, 1, 1)

        self.prefix_entry = Gtk.Entry()
        self.prefix_entry.set_text(DEFAULT_PREFIX)
        self.prefix_entry.set_width_chars(30)
        self.prefix_entry.connect("changed", self._on_prefix_changed)
        left_grid.attach(self.prefix_entry, 1, 1, 1, 1)

        # Scene Suffix
        scene_label = Gtk.Label(label="Scene Suffix:")
        scene_label.set_xalign(0)
        left_grid.attach(scene_label, 0, 2, 1, 1)

        self.scene_entry = Gtk.Entry()
        self.scene_entry.set_text("001")
        self.scene_entry.set_width_chars(5)
        self.scene_entry.set_max_length(3)
        self.scene_entry.set_input_purpose(Gtk.InputPurpose.DIGITS)
        self.scene_entry.connect("insert-text", self._on_scene_insert_text)
        self.scene_entry.connect("changed", self._on_scene_changed)
        left_grid.attach(self.scene_entry, 1, 2, 1, 1)

        # Frame Number
        frame_label = Gtk.Label(label="Frame Number:")
        frame_label.set_xalign(0)
        left_grid.attach(frame_label, 0, 3, 1, 1)

        self.frame_entry = Gtk.Entry()
        self.frame_entry.set_text("000001")
        self.frame_entry.set_width_chars(8)
        self.frame_entry.set_max_length(6)
        self.frame_entry.set_input_purpose(Gtk.InputPurpose.DIGITS)
        self.frame_entry.connect("insert-text", self._on_frame_insert_text)
        self.frame_entry.connect("changed", self._on_frame_changed)
        left_grid.attach(self.frame_entry, 1, 3, 1, 1)

        left_vbox.append(left_grid)
        columns_hbox.append(left_vbox)

        # Right column: File Type
        right_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        right_vbox.set_hexpand(False)

        right_header = Gtk.Label(label="File Type")
        right_header.set_xalign(0)
        right_vbox.append(right_header)

        # Radio buttons and checkbox stacked vertically
        self.filetype_raw_radio = Gtk.CheckButton(label="RAW 4064×3040")
        self.filetype_raw_radio.set_active(True)
        self.filetype_raw_radio.connect("toggled", self._on_filetype_changed)
        right_vbox.append(self.filetype_raw_radio)

        self.filetype_debayered_radio = Gtk.CheckButton(label="RGB 2028×1520")
        self.filetype_debayered_radio.set_group(self.filetype_raw_radio)
        self.filetype_debayered_radio.connect("toggled", self._on_filetype_changed)
        right_vbox.append(self.filetype_debayered_radio)

        self.crop_rgb_checkbox = Gtk.CheckButton(label="Crop & Stabilize")
        self.crop_rgb_checkbox.set_active(False)
        self.crop_rgb_checkbox.set_sensitive(False)  # greyed out when Raw is selected
        self.crop_rgb_checkbox.connect("toggled", self._on_crop_rgb_toggled)
        right_vbox.append(self.crop_rgb_checkbox)

        # Spacer to push Save button to the bottom
        spacer = Gtk.Box()
        spacer.set_vexpand(True)
        right_vbox.append(spacer)

        # Save button
        self.save_button = Gtk.Button(label="Save")
        self.save_button.connect("clicked", self._on_save_clicked)
        right_vbox.append(self.save_button)

        columns_hbox.append(right_vbox)

        vbox.append(columns_hbox)

        return frame

    def _on_filepath_changed(self, entry):
        """Publish the save directory as it is edited."""
        write_string(StringIndex.FILE_PATH, entry.get_text())

    def _on_prefix_changed(self, entry):
        """Publish the filename prefix as it is edited."""
        write_string(StringIndex.METADATA_1, entry.get_text())

    def _on_scene_changed(self, entry):
        """Publish the scene suffix as it is edited."""
        write_string(StringIndex.METADATA_2, entry.get_text())

    def _on_frame_changed(self, entry):
        """Publish the frame number as it is edited."""
        write_string(StringIndex.USER_COMMENT, entry.get_text())

    def _on_browse_clicked(self, button):
        """Open the folder chooser for the save destination."""
        dialog = Gtk.FileChooserDialog(
            title="Select Save Location",
            action=Gtk.FileChooserAction.SELECT_FOLDER,
        )

        dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("_Select", Gtk.ResponseType.ACCEPT)

        dialog.set_modal(True)
        dialog.set_transient_for(button.get_root())

        dialog.connect("response", self._on_folder_selected)

        dialog.show()

    def _on_folder_selected(self, dialog, response):
        """Put the chosen folder into the filepath entry, which publishes it to shared memory."""
        if response == Gtk.ResponseType.ACCEPT:
            folder = dialog.get_file()
            if folder:
                path = folder.get_path()
                self.filepath_entry.set_text(path)
                log(LogTag.GUI, f"File Save Panel - Selected folder: {path}")

        dialog.destroy()

    def _on_filetype_changed(self, radio_button):
        """Switch between sensor raw and debayered RGB output, enabling crop only for RGB."""
        if self.filetype_raw_radio.get_active():
            self.flags[FlagIndex.SAVE_RAW] = True
            self.crop_rgb_checkbox.set_active(False)
            self.crop_rgb_checkbox.set_sensitive(False)
            log(LogTag.GUI, "File type changed to Sensor Raw")
        else:
            self.flags[FlagIndex.SAVE_RAW] = False
            self.crop_rgb_checkbox.set_sensitive(True)
            self.crop_rgb_checkbox.set_active(True)
            log(LogTag.GUI, "File type changed to RGB")

    def _on_crop_rgb_toggled(self, checkbox):
        """Publish whether crop and stabilisation should be applied on save."""
        is_active = checkbox.get_active()
        self.flags[FlagIndex.SAVE_APPLY_CROP] = is_active
        log(LogTag.GUI, f"Crop & Stabilize {'enabled' if is_active else 'disabled'}")

    def _on_save_clicked(self, button):
        """Request a save; the settings themselves were already published as they were edited."""
        filetype = "raw" if self.filetype_raw_radio.get_active() else "debayered_half"

        # Settings are already in shared memory from widget handlers
        # Just log what we're saving
        filepath = self.get_filepath()
        prefix = self.get_prefix()
        scene = self.get_scene_suffix()
        frame = self.get_frame_number()

        log(LogTag.GUI, f"Save — type:{filetype} scene:{scene} frame:{frame} path:{filepath}")

        # Trigger save
        self.flags[FlagIndex.SAVE_REQUEST] = True

        log(LogTag.GUI, "Save request sent to file saver module")

    def _on_scene_insert_text(self, entry, text, length, position):
        """Reject non-digit input in the scene field."""
        if not text.isdigit():
            entry.stop_emission_by_name("insert-text")
            return True
        return False

    def _on_frame_insert_text(self, entry, text, length, position):
        """Reject non-digit input in the frame field."""
        if not text.isdigit():
            entry.stop_emission_by_name("insert-text")
            return True
        return False

    def get_filepath(self):
        """Return the save directory shown in the entry."""
        return self.filepath_entry.get_text()

    def get_prefix(self):
        """Return the filename prefix shown in the entry."""
        return self.prefix_entry.get_text()

    def get_scene_suffix(self):
        """Return the scene suffix shown in the entry."""
        return self.scene_entry.get_text()

    def get_frame_number(self):
        """Return the frame number shown in the entry."""
        return self.frame_entry.get_text()

    def update_display(self):
        """Refresh the frame number from shared memory once a save has completed."""
        try:
            # Only update if a save just completed
            if self.flags[FlagIndex.SAVE_COMPLETE]:
                # Read updated frame number from shared memory
                frame_str = read_string(StringIndex.USER_COMMENT)

                if frame_str:
                    # Block signal handlers during programmatic update to avoid validation warnings
                    self.frame_entry.handler_block_by_func(self._on_frame_insert_text)
                    self.frame_entry.handler_block_by_func(self._on_frame_changed)
                    self.frame_entry.set_text(frame_str)
                    self.frame_entry.handler_unblock_by_func(self._on_frame_changed)
                    self.frame_entry.handler_unblock_by_func(self._on_frame_insert_text)

                # Clear the flag so we don't keep updating
                self.flags[FlagIndex.SAVE_COMPLETE] = False

            return True
        except Exception as e:
            # Silently ignore errors during update
            return True