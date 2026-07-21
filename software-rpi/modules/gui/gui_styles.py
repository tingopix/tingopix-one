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
GUI theme and style tokens

Applies the application-wide dark theme to the GTK4 display. Sets GTK's dark
theme preference and loads the CSS that defines the panel frame styling, the
amber #b2770b frame-title accent, and the shared text colour and font size used
across every widget type in the interface.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    Library-style module, called directly. mod_gui calls apply_dark_theme() once
    during window creation, before any panel frames are built.
"""
import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gdk', '4.0')
from gi.repository import Gtk, Gdk
from tpx_logger import log, LogTag


def apply_dark_theme():
    """Apply the dark theme and the application-wide CSS to the default display."""
    # Force dark theme preference
    settings = Gtk.Settings.get_default()
    if settings:
        settings.set_property("gtk-application-prefer-dark-theme", True)

    css_provider = Gtk.CssProvider()

    css_provider.load_from_data(b"""
        window {
            background-color: #1e1e1e;
            color: #b0b0b0;
        }
        frame {
            background-color: #2d2d2d;
            border: 1px solid #404040;
            border-radius: 6px;
        }
        frame > label {
            color: #b2770b;
            font-weight: bold;
            font-size: 11pt;
        }
        switch:checked {
            background-color: #1a1a1a;
        }
        scale
        {
            color: #808080;
            font-size: 10pt;
        }
        scale trough {
            background-color: #1a1a1a;
        }
        scale trough highlight {
            background-color: #808080;
            min-width: 4px;
            min-height: 4px;
        }
        scale slider {
            background-color: #404040;
            border: 2px solid #b2770b;
            box-shadow: none;
            min-width: 18px;
            min-height: 18px;
            border-radius: 9px;
        }

        label
        {
            color: #808080;
            font-size: 10pt;
        }

        button
        {
            color: #808080;
            font-size: 10pt;
        }
                                
        entry
        {
            color: #808080;
            font-size: 10pt;
        }
        checkbutton
        {
            color: #808080;
            font-size: 10pt;
        }
        radiobutton
        {
            color: #808080;
            font-size: 10pt;
        }
                                
        menubutton
        {
            color: #808080;
            font-size: 10pt;
        }
        spinbutton
        {
            color: #808080;
            font-size: 10pt;
        }
        levelbar
        {
            color: #808080;
            font-size: 10pt;
        }
    """)

    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_USER
    )

    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css_provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    log(LogTag.GUI, "Dark theme applied")