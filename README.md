# Tingopix

An open DIY film scanner for 8mm, Regular 8, Super 8, and 16mm.
Sensel-native, archival-first.

Website: https://tingopix.github.io
Repository: https://github.com/tingopix/tingopix-one

---

## About

Tingopix is a DIY film scanner built around a Raspberry Pi and the Raspberry Pi
HQ camera (IMX477). It is designed for archival-quality results at accessible
cost, trading scanning speed for image quality rather than the other way around.

Frames are captured at the sensor's native resolution and written as linear
16-bit TIFF. No log curve, tone mapping or display transform is baked into the
output — the archival file stays linear, and colour transforms belong further
down the pipeline in whatever grading application you use. Colour correction
matrices and per-channel digital gains are applied at development time by the
post-scan processing tool, not burned into the capture.

This is a build-it-yourself project, not a product. It assumes you are willing
to assemble hardware, edit a configuration file, and read a log.

## Status

- Software is functional and installable.
- Third-party hardware is supported through a custom adapter interface.
  Implement the documented methods in `CustomAdapter` (`software-rpi/mod_serial.py`)
  and set `HARDWARE_MODE = "CUSTOM"` in `software-rpi/tpx_config.py`. The capture
  pipeline, GUI and shared-memory layer are unchanged; only the transport and
  light-control calls are routed to your hardware.
- Hardware and firmware documentation are in progress.

## Repository structure

```
tingopix-one/
├── software-rpi/     Raspberry Pi scanner application (GTK4 GUI, capture,
│                     shared-memory IPC). Has its own requirements.txt.
├── postprocess/      Standalone offline post-scan processing tool. Develops
│                     raw Bayer frames to viewable images. Has its own
│                     requirements.txt, intended for off-Pi installs.
└── install.sh        One-line installer for the Raspberry Pi.
```

`hardware/` and `firmware/` will be added when their documentation is ready.

> **Note for Pi users:** `postprocess/requirements.txt` is for standalone
> installs on a separate machine. Do **not** pip-install it into the scanner's
> virtual environment — see [postprocess/README.md](postprocess/README.md).

## Installation

On a Raspberry Pi running Raspberry Pi OS:

```bash
curl -fsSL https://raw.githubusercontent.com/tingopix/tingopix-one/main/install.sh | bash
```

The installer runs as a normal user and invokes `sudo` only for `apt` and for
writes to `/usr/local/bin`. It:

- installs two apt packages (`gir1.2-gtk-4.0`, `python3-gi-cairo`) — everything
  else it needs ships with Raspberry Pi OS;
- clones the repository to `~/tingopix-one`;
- creates a virtual environment at `software-rpi/env-tpx` with
  `--system-site-packages`, so the system `picamera2`, `numpy` and GTK bindings
  are inherited rather than duplicated;
- pip-installs only `numba` and `tifffile` into that environment, then verifies
  that `numpy` still resolves to the system build;
- writes three commands to `/usr/local/bin` — `tingopix`, `tingopix-update` and
  `tingopix-postprocess` — plus a desktop entry and icon.

The installer is idempotent. Re-running it is safe, and is also how you repair a
broken install or pick up changes to the `/usr/local/bin` commands.

## Usage

Launch from the applications menu, or from a terminal:

```bash
tingopix
```

**The first launch takes about 100 seconds before the window appears.** Numba
compiles the processing kernels on first run; subsequent launches take roughly
8 seconds. This is why the desktop entry opens a terminal — the compile progress
and an ETA are printed there. It has not hung.

To update:

```bash
sudo tingopix-update
```

This pulls the latest `main` and refreshes pip dependencies. It does not
regenerate the `/usr/local/bin` commands; re-run the installer for that.

## Post-scan processing

`postprocess/` contains a standalone tool that develops the raw Bayer TIFFs
written by the scanner: black level correction, per-channel digital gain, an
optional colour correction matrix, and demosaicing, written out as 16-bit TIFF
or OpenEXR.

On the Pi, the installer provides it as a command for spot-checking individual
frames:

```bash
tingopix-postprocess in.tif out.tif --ccm
```

The tool is a single self-contained file with no dependency on the scanner
software, and it runs on Linux, Windows and macOS. **For anything beyond
checking a frame or two, install it on a separate machine.** A workstation is
substantially faster than a Pi for batch development, and OpenEXR output is not
available from the Pi installer — the `pyexr` dependency needs system OpenEXR
libraries and builds from source on aarch64, so it is deliberately not bundled.

See [postprocess/README.md](postprocess/README.md) for installation, the full
command-line reference, and notes on the demosaic kernels.

## License

Tingopix is released under three licenses:

- **Software and firmware** — MIT
- **Hardware** — CERN-OHL-P v2 (when published)
- **Documentation** — CC BY 4.0

See [LICENSE](LICENSE) for the full text and details of what each covers.
