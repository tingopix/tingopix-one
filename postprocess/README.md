# Tingopix — Post-Scan Processing

Offline development of raw Bayer frames captured by the Tingopix scanner.

This is a standalone tool. It is a single self-contained Python file with no
dependency on the scanner software, and it runs on Linux, Windows and macOS as
well as on the Raspberry Pi itself.

---

## What it does

The scanner writes raw sensor data — a single-channel Bayer mosaic, with no
demosaicing, white balance or colour transform applied. This tool develops those
frames into viewable RGB images.

The pipeline, in the order it runs:

1. **Load and validate** the raw frame. Geometry is enforced, not assumed.
2. **Demosaic** — bilinear by default, or variable-number-of-gradients with `--vng`.
3. **Black level correction** — subtracted from every channel.
4. **Digital gain** — per-channel red and blue multipliers. Green is always 1.0.
5. **Colour correction matrix** — optional, applied only with `--ccm`.
6. **Crop stabilisation** — optional, anchored to the detected sprocket hole.
7. **Write** — 16-bit TIFF, or OpenEXR.

Two things about this order are worth knowing:

- **Demosaicing happens first**, before black level and colour correction. The
  interpolation works on the raw mosaic values.
- **Black level correction is not gated by `--ccm`.** It runs on every frame
  regardless. `--ccm` controls *only* whether the colour matrix is applied. To
  disable black level subtraction entirely, pass `--blc 0`.

### Input requirements

Input must be exactly **4064 × 3040**, single-channel, 16-bit TIFF, in **GBRG**
Bayer phase — the full-sensor readout of the Raspberry Pi HQ camera (IMX477).

This is checked on load and a mis-shaped frame is rejected rather than processed.
That is deliberate: the crop arithmetic and the Bayer phase assumptions both
depend on the exact geometry, and a wrong-sized frame would debayer into output
that looks plausible but is incorrect.

### Embedded metadata

The capture side writes sprocket-tracking metadata and digital gains into the
**last column of the frame, rows 0–6**:

| Row | Contents |
|-----|----------|
| 0 | Film position |
| 1 | Film edge |
| 2 | Marker position |
| 3 | Crop height |
| 4 | Capture mode — `0x0000` RGB, `0x0001` ALL |
| 5 | Red digital gain (× 1000) |
| 6 | Blue digital gain (× 1000) |

A sentinel of `0xFFFF` in rows 0 or 1 means the capture found no sprocket hole;
`--crop-stabilize` then logs a message and passes the frame through uncropped.

**Capture mode** records how the frame was shot, and determines whether the
embedded gains carry balance information:

- **ALL** — every channel captured simultaneously in a single exposure. The
  channels are unbalanced by nature, so the embedded gains are real values and
  are what brings them into balance.
- **RGB** — each channel captured separately, with its own exposure and light
  setting. Balance is achieved optically at capture time, so no digital gain
  applies and the embedded gains are 1.0.

The mode is reported on the metadata line for each frame. It is informational —
this tool does not change its processing based on it.

Do not crop or otherwise strip the last column before running this tool, or the
metadata is lost. It is read automatically; `--use-dig-gains` controls whether
the embedded gains are *applied*.

---

## Installation

### Standalone (Linux, Windows, macOS) — recommended

```bash
cd postprocess
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then run it directly:

```bash
python debayer_vng.py in.tif out.tif --ccm
```

This is the setup to use for developing more than a handful of frames. A desktop
or laptop is substantially faster than a Raspberry Pi at this work, and OpenEXR
output is straightforward to enable here (see below).

### On the Raspberry Pi

Nothing to install. The Tingopix installer already provides the tool as a
command:

```bash
tingopix-postprocess in.tif out.tif --ccm
```

The scanner's virtual environment already supplies `numpy`, `numba` and
`tifffile`, so no additional packages are needed.

> **Do not run `pip install -r postprocess/requirements.txt` inside the scanner's
> virtual environment.** That environment is created with
> `--system-site-packages` and relies on the system `numpy` that `picamera2` is
> built against. This file pins `numpy>=2.0`, and installing it into the venv
> would shadow the system build and break the camera. The installer deliberately
> never points at this file, and it includes a guard that aborts if `numpy` ends
> up resolving into the venv.

Throughout the rest of this document, `tingopix-postprocess` and
`python debayer_vng.py` are interchangeable — use whichever suits where you are
running it.

---

## Usage

Develop a single frame with the colour matrix applied:

```bash
tingopix-postprocess in.tif out.tif --ccm
```

Develop with a specific colour temperature and the gains recorded at capture:

```bash
tingopix-postprocess in.tif out.tif --ccm --color-temp 3200 --use-dig-gains
```

Develop a whole directory:

```bash
tingopix-postprocess raw/ developed/ --batch --ccm --use-dig-gains
```

Higher-quality demosaic, cropped to the sprocket hole:

```bash
tingopix-postprocess in.tif out.tif --ccm --vng --crop-stabilize
```

---

## Command-line options

| Option | Argument | Default | Description |
|--------|----------|---------|-------------|
| `input` | path | — | Input TIFF file, or input directory with `--batch`. |
| `output` | path | — | Output file, or output directory with `--batch`. |
| `--batch` | — | off | Treat `input` and `output` as directories and process every `.tif` / `.tiff` found. |
| `--blc` | `LEVEL` | `4096` | Black level subtracted from every channel, in 16-bit counts. Runs whether or not `--ccm` is given. Pass `0` to disable. Must be 0–65535. |
| `--ccm` | — | off | Apply the colour correction matrix. Gates only the matrix step, not black level correction. |
| `--color-temp` | `KELVIN` | `5600` | Colour temperature for the CCM. The nearest calibrated matrix is used, with no interpolation. |
| `--r-gain` | `GAIN` | — | Red channel gain. Overrides any embedded gain. Must be positive. |
| `--b-gain` | `GAIN` | — | Blue channel gain. Overrides any embedded gain. Must be positive. |
| `--use-dig-gains` | — | off | Read and apply the per-channel digital gains embedded in the raw frame. |
| `--vng` | — | off | Use variable-number-of-gradients demosaicing instead of bilinear. Slower, but holds edges better and reduces colour fringing. |
| `--crop-stabilize` | — | off | Crop the frame to the detected sprocket hole. Skipped with a message if no sprocket was found. |
| `--exr` | — | off | Write OpenEXR 16-bit half-float instead of TIFF. Requires `pyexr`. |
| `--exr-full` | — | off | Write OpenEXR 32-bit float. Implies `--exr`. Requires `pyexr`. |

### Gain precedence

Gains resolve in this order, each overriding the previous:

1. Default — red and blue both `1.0`.
2. `--use-dig-gains` — embedded values from the frame. If they cannot be read or
   are not positive, a warning is logged and the values fall back to `1.0`.
3. `--r-gain` / `--b-gain` — explicit command-line values, applied last.

Because the override is per channel rather than all-or-nothing, two useful cases
fall out:

**Adjust the gains recorded on an ALL capture.** Take the embedded values as the
starting point and replace just one channel:

```bash
tingopix-postprocess in.tif out.tif --ccm --use-dig-gains --r-gain 1.8
```

The blue gain stays as captured; only red is overridden.

**Trim red or blue on an RGB capture.** These frames were balanced optically, so
their embedded gains are 1.0 and there is no reason to pass `--use-dig-gains`.
The gain flags still work on their own, starting from a 1.0 baseline:

```bash
tingopix-postprocess in.tif out.tif --ccm --r-gain 1.05
```

A gain above `5.0` logs a warning but is not rejected.

### Calibrated colour temperatures

`--color-temp` snaps to the nearest of these measured values. No interpolation is
performed between them:

```
2000  2200  2400  2600  2800  3000  3200  3400  3600
4100  4600  5100  5600  6100  6600  7100  7600  8100  8600
```

The temperature actually used is logged, so you can confirm which calibration
point was applied.

---

## Demosaic kernels

**Bilinear** is the default. It is fast and adequate for checking exposure,
framing and colour.

**VNG** (`--vng`) examines gradients in several directions and interpolates along
the one with least variation. It holds hard edges better and reduces colour
fringing, at a noticeable cost in speed.

### The VNG threshold knob

VNG decides which gradient directions to trust using a threshold derived from the
smallest gradient found, scaled by a constant:

```python
VNG_THRESHOLD_SCALE = 1.2   # in debayer_vng.py
```

**This constant was tuned against synthetic test images, not scanned film.** On
the test material it wins clearly on hard edges and is marginally worse on smooth
or grainy content — and grain is exactly what real film has.

If developed frames look over-sharpened or the grain looks crunchy, raising this
value toward `1.5` widens the band of directions considered acceptable, which
softens the discrimination. This is a single-line edit in `debayer_vng.py`. It is
worth trying on your own footage before concluding VNG is not for you.

---

## Performance

On a Raspberry Pi, expect this to be slow — it is a full-resolution
floating-point pipeline on a 12-megapixel frame, and `--vng` is considerably
slower than bilinear. The Pi command exists for spot-checking a frame or two, not
for developing a roll.

Numba compiles the processing kernels on first run, which adds a one-off delay of
roughly half a minute before the first frame is processed. The result is cached
on disk, so subsequent runs start immediately. The `tingopix-postprocess` wrapper
changes into the `postprocess/` directory before running precisely so that this
cache lands in one place and is reused regardless of where you invoke it from.

**For batch work, use a separate machine.** That is the main reason this tool is
standalone rather than integrated into the scanner software.

---

## OpenEXR output

`--exr` writes 16-bit half-float and `--exr-full` writes 32-bit float, both with
lossless PIZ compression.

The reason to use EXR is range. The colour correction matrix can produce values
below 0.0 and above 1.0, and 16-bit integer TIFF clips both. EXR preserves them,
which matters if the frames are heading into a grading pipeline where that
headroom will be recovered.

EXR output requires `pyexr`, which is **deliberately not included** in
`requirements.txt` or installed by the Pi installer. It needs system OpenEXR
libraries present at build time and has no prebuilt wheel for aarch64, so on a
Raspberry Pi it compiles from source — a slow and failure-prone step to put in a
default install path.

To enable it:

```bash
# Debian / Ubuntu / Raspberry Pi OS
sudo apt install libopenexr-dev
pip install pyexr
```

```bash
# macOS
brew install openexr
pip install pyexr
```

On Windows, install `pyexr` into your environment with `pip`; a prebuilt wheel is
generally available.

Requesting `--exr` without `pyexr` installed fails immediately with an
explanatory message rather than partway through a batch.

---

## Logging

Console output matches the format used by the scanner software, so logs from both
read consistently:

```
[14:32:07] [POST ] Processed: frame_0001.tif -> frame_0001.tif
[14:32:07] [WARN ] [POST ] Could not extract embedded gains, falling back to 1.0 / 1.0
```

Warnings and errors go to stderr, so they remain visible if you redirect stdout to
a file.

The logging format is reimplemented locally rather than imported from the scanner
software. That is intentional — it is what keeps this file usable on its own,
away from the Pi.

---

## License

MIT. See [LICENSE](LICENSE) in this directory, and the
[root LICENSE](../LICENSE) for the project-wide licensing split.
