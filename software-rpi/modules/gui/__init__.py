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
GUI package

Collects the panel classes and the theme helper that make up the interface, so
the GUI process can import them from one place. Each panel builds its own frame
and refreshes itself; this module only defines the package surface.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    mod_gui (imports every panel listed here to assemble the main window)
"""

from modules.gui.gui_base_panel import BasePanel
from modules.gui.gui_aoi_panel import AOIPanel
from modules.gui.gui_rgb_panel import RGBPanel
from modules.gui.gui_waveform_panel import WaveformPanel
from modules.gui.gui_filesave_panel import FileSavePanel
from modules.gui.gui_scancontrol_panel import ScanControlPanel
from modules.gui.gui_scanbuttons_panel import ScanButtonsPanel
from modules.gui.gui_gain_panel import GainPanel
from modules.gui.gui_system_panel import SystemPanel
from modules.gui.gui_styles import apply_dark_theme

__all__ = [
    'BasePanel',
    'AOIPanel',
    'RGBPanel',
    'WaveformPanel',
    'FileSavePanel',
    'ScanControlPanel',
    'ScanButtonsPanel',
    'GainPanel',
    'SystemPanel',
    'apply_dark_theme'
]