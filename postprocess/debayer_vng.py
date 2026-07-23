#!/usr/bin/env python3
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
Offline debayering for GBRG Bayer-pattern raw frames.

Standalone development pipeline for scanned frames captured with the
Raspberry Pi HQ camera (IMX477), where standard ISP metadata is not
available. Reads 16-bit TIFF raw Bayer data and applies black level
correction, per-channel digital gain, an optional colour correction
matrix and debayering, writing 16-bit TIFF or OpenEXR.

Input frames must be exactly 4064x3040 single-channel 16-bit TIFF, as
written by the capture side. Sprocket-tracking metadata and per-channel
digital gains are read from the frame's last column.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -  (standalone offline tool; no runtime coupling to the scanner)

Examples:
    python debayer_vng.py frame.tif out.tif
    python debayer_vng.py frame.tif out.tif --ccm --color-temp 3200
    python debayer_vng.py raw/ developed/ --batch --exr --use-dig-gains

Optional dependencies:
    pyexr    required only for --exr / --exr-full output
"""

import numpy as np
import tifffile
try:
    import pyexr
except ImportError:
    pyexr = None
from numba import njit, prange
import argparse
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# Console logging
# ============================================================================
# Format matches tpx_logger from the main TingoPix project so offline output
# reads the same as the scanner's. Reimplemented locally rather than imported
# to keep this file standalone — it is distributed as a single script.

TAG_WIDTH = 5
LOG_TAG = "POST"


def log(message: str, level: str = "INFO") -> None:
    """Print a timestamped, tagged log line.

    Args:
        message: The message string (already formatted by the caller).
        level:   "INFO" (default), "WARN", or "ERR". WARN and ERR are
                 written to stderr so they survive output redirection.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    tag_str = f"[{LOG_TAG:<{TAG_WIDTH}}]"

    if level == "WARN":
        print(f"[{ts}] [WARN ] {tag_str} {message}", file=sys.stderr, flush=True)
    elif level == "ERR":
        print(f"[{ts}] [ERR  ] {tag_str} {message}", file=sys.stderr, flush=True)
    else:
        print(f"[{ts}] {tag_str} {message}", flush=True)


# ============================================================================
# Constants & CCM calibration table
# ============================================================================
DEFAULT_BLACK_LEVEL = 4096

# Colour temperature used when --color-temp is not given. Must be one of the
# calibrated values in CCM_TEMPS below; anything else snaps to the nearest.
DEFAULT_CCM_TEMPERATURE = 5600

# Expected raw frame geometry (IMX477 full sensor readout). Enforced on input:
# the crop maths and the GBRG phase assumptions both depend on these.
FRAME_WIDTH = 4064
FRAME_HEIGHT = 3040

# Rows of the final image column used to carry embedded metadata.
META_ROWS_REQUIRED = 7

CCM_TEMPS = np.array([
    2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
    4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600
], dtype=np.int32)

CCM_MATRICES = np.array([
    [[1.581388, -0.352937, -0.273788], [-0.434730, 1.579263, -0.121026], [0.232229, -1.438267, 2.138643]],
    [[1.632205, -0.459323, -0.213735], [-0.397072, 1.587787, -0.172494], [0.207538, -1.266067, 2.005654]],
    [[1.676661, -0.544710, -0.168386], [-0.365985, 1.592224, -0.212709], [0.183396, -1.133916, 1.908934]],
    [[1.716198, -0.615259, -0.133110], [-0.339721, 1.594489, -0.245398], [0.161558, -1.029868, 1.835785]],
    [[1.751931, -0.674868, -0.105152], [-0.317170, 1.595582, -0.272740], [0.142309, -0.946098, 1.778709]],
    [[1.784672, -0.726124, -0.082747], [-0.297565, 1.596043, -0.296104], [0.125464, -0.877343, 1.733036]],
    [[1.815009, -0.770811, -0.064695], [-0.280347, 1.596169, -0.316404], [0.110715, -0.819977, 1.695721]],
    [[1.843367, -0.810206, -0.050135], [-0.265093, 1.596129, -0.334276], [0.097748, -0.771430, 1.664707]],
    [[1.870058, -0.845252, -0.038426], [-0.251479, 1.596018, -0.350177], [0.086285, -0.729842, 1.638553]],
    [[1.898870, -0.891128, -0.018849], [-0.214871, 1.599237, -0.394055], [0.082515, -0.717892, 1.626701]],
    [[1.960355, -0.962434, -0.001712], [-0.194446, 1.597849, -0.416728], [0.063103, -0.648379, 1.583461]],
    [[2.014681, -1.019593, 0.007728], [-0.177520, 1.597708, -0.436609], [0.047413, -0.595033, 1.551292]],
    [[2.062652, -1.065839, 0.011886], [-0.163192, 1.598363, -0.454221], [0.034658, -0.553545, 1.526903]],
    [[2.104986, -1.103598, 0.012504], [-0.150908, 1.599470, -0.469841], [0.024218, -0.520892, 1.508127]],
    [[2.142499, -1.134760, 0.010730], [-0.140218, 1.600822, -0.483792], [0.015521, -0.494636, 1.493331]],
    [[2.175803, -1.160756, 0.007453], [-0.130857, 1.602265, -0.496233], [0.008227, -0.473308, 1.481534]],
    [[2.205529, -1.182666, 0.003202], [-0.122572, 1.603726, -0.507397], [0.002013, -0.455659, 1.471940]],
    [[2.232225, -1.201367, -0.001623], [-0.115180, 1.605154, -0.517456], [-0.003338, -0.440859, 1.464025]],
    [[2.256082, -1.217321, -0.006723], [-0.108603, 1.606515, -0.526473], [-0.007953, -0.428400, 1.457465]]
], dtype=np.float32)

# ============================================================================
# CCM selection & embedded frame metadata
# ============================================================================

def get_ccm_matrix(color_temp):
    """Select the calibrated CCM nearest to a colour temperature.

    No interpolation is performed: only the measured matrices in CCM_TEMPS
    are ever used. Returns (matrix, selected_temp) so callers can report
    which calibration point was actually applied.
    """
    diffs = np.abs(CCM_TEMPS.astype(np.float32) - float(color_temp))
    idx = int(np.argmin(diffs))
    return CCM_MATRICES[idx], int(CCM_TEMPS[idx])


def extract_metadata(raw_frame):
    """Extract sprocket-tracking metadata from the frame's last column.

    Returns (film_pos, film_edge, marker_pos, crop_h, has_sprocket), or None
    if the values could not be read. A sentinel of 0xFFFF in film_pos or
    film_edge means the capture found no sprocket hole, in which case the
    tuple is still returned but has_sprocket is False.
    """
    if raw_frame.shape[0] < META_ROWS_REQUIRED:
        log(f"Frame has {raw_frame.shape[0]} rows, "
            f"need {META_ROWS_REQUIRED} to read metadata", level="WARN")
        return None

    try:
        film_pos_raw = int(raw_frame[0, -1])
        film_edge_raw = int(raw_frame[1, -1])
        marker_pos_raw = int(raw_frame[2, -1])
        crop_h_raw = int(raw_frame[3, -1])
    except (IndexError, ValueError) as e:
        log(f"Could not read embedded metadata: {e}", level="WARN")
        return None

    # Sentinel 0xFFFF means the capture side detected no sprocket.
    has_sprocket = (film_pos_raw != 65535) and (film_edge_raw != 65535)

    # Stored as unsigned; reinterpret as signed for real positions.
    film_pos = int(np.int16(film_pos_raw)) if has_sprocket else film_pos_raw
    film_edge = int(np.int16(film_edge_raw)) if has_sprocket else film_edge_raw
    marker_pos = int(np.int16(marker_pos_raw))
    crop_h = int(np.int16(crop_h_raw))

    log(f"Metadata: film_pos={film_pos_raw}, film_edge={film_edge_raw}, "
        f"marker_pos={marker_pos_raw}, crop_h={crop_h_raw}, "
        f"has_sprocket={has_sprocket}")

    return film_pos, film_edge, marker_pos, crop_h, has_sprocket


def extract_digital_gains(raw_frame):
    """Extract per-channel digital gains from the frame's last column.

    Row 5 holds the red gain and row 6 the blue gain, both stored as the
    gain multiplied by 1000. Green is implicitly 1.0. Returns (r_gain,
    b_gain), or None if the values could not be read.
    """
    if raw_frame.shape[0] < META_ROWS_REQUIRED:
        return None

    try:
        r_gain = float(raw_frame[5, -1]) / 1000.0
        b_gain = float(raw_frame[6, -1]) / 1000.0
    except (IndexError, ValueError) as e:
        log(f"Could not read embedded gains: {e}", level="WARN")
        return None

    if r_gain <= 0.0 or b_gain <= 0.0:
        log(f"Embedded gains are not positive "
            f"(R={r_gain:.3f}, B={b_gain:.3f}), ignoring them", level="WARN")
        return None

    return r_gain, b_gain


def calculate_crop_params(film_pos, marker_pos, crop_h, film_edge, frame_h):
    """Derive crop bounds that stabilise the frame against the sprocket hole.

    The vertical offset is the difference between the detected sprocket
    position and the reference marker; the horizontal crop is anchored to the
    detected film edge. Bounds are clamped so the crop always stays inside
    the frame. Returns (left, right, top, bottom).
    """
    aoi_w, frame_w = 256, FRAME_WIDTH
    crop_w = frame_w - aoi_w
    v_shift = film_pos - marker_pos
    top = (frame_h - crop_h) // 2 + v_shift
    bot = top + crop_h
    if top < 0: top, bot = 0, crop_h
    elif bot > frame_h: bot, top = frame_h, frame_h - crop_h
    right = film_edge
    left = film_edge - crop_w
    if left < 0: left, right = 0, crop_w
    elif right > frame_w: right, left = frame_w, frame_w - crop_w
    return left, right, top, bot

# ============================================================================
# Debayer & colour pipeline (Numba)
# ============================================================================

@njit(inline='always')
def interp_green_simple_f(raw_f, r, c, h, w):
    """Average the four orthogonal neighbours of a non-green pixel.

    Neighbours at the frame edge are omitted rather than mirrored, so edge
    pixels average over fewer samples.
    """
    v, cnt = 0.0, 0.0
    if r > 0: v += raw_f[r-1, c]; cnt += 1.0
    if r < h-1: v += raw_f[r+1, c]; cnt += 1.0
    if c > 0: v += raw_f[r, c-1]; cnt += 1.0
    if c < w-1: v += raw_f[r, c+1]; cnt += 1.0
    return v / cnt if cnt > 0.0 else 0.0

@njit(inline='always')
def interp_rb_bilinear_f(raw_f, r, c, h, w, target_r):
    """Average the red or blue samples in the 3x3 neighbourhood.

    In GBRG, red sits at odd rows and even columns, blue at even rows and
    odd columns. target_r selects which of the two to gather.
    """
    v, cnt = 0.0, 0.0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            curr_r, curr_c = r+dr, c+dc
            if 0 <= curr_r < h and 0 <= curr_c < w:
                if target_r:
                    if (curr_r & 1) == 1 and (curr_c & 1) == 0: v += raw_f[curr_r, curr_c]; cnt += 1.0
                else:
                    if (curr_r & 1) == 0 and (curr_c & 1) == 1: v += raw_f[curr_r, curr_c]; cnt += 1.0
    return v / cnt if cnt > 0.0 else 0.0

@njit(parallel=True, cache=True, fastmath=True)
def vng_debayer_gbrg_float(raw_f, rgb_f):
    """Debayer a normalised GBRG frame in place into rgb_f.

    Pixel type per position: 0 = green, 1 = red, 2 = blue. The sampled
    channel is copied through unchanged and the other two are interpolated
    from the surrounding neighbourhood.
    """
    h, w = raw_f.shape
    for row in prange(h):
        r_even = (row & 1) == 0
        for col in range(w):
            c_even = (col & 1) == 0
            p_type = 0 
            if r_even: p_type = 0 if c_even else 2
            else: p_type = 1 if c_even else 0
            cur = raw_f[row, col]
            rgb_f[row, col, 0] = interp_rb_bilinear_f(raw_f, row, col, h, w, True) if p_type != 1 else cur
            rgb_f[row, col, 1] = cur if p_type == 0 else interp_green_simple_f(raw_f, row, col, h, w)
            rgb_f[row, col, 2] = interp_rb_bilinear_f(raw_f, row, col, h, w, False) if p_type != 2 else cur

@njit(parallel=True, cache=True)
def apply_blc_ccm_float(rgb_f, blc_f, apply_ccm, ccm, r_gain, b_gain):
    """Apply black level, per-channel gain and optional CCM in place.

    Order matters: black level is subtracted first so the gains and the
    matrix operate on true zero-referenced values. Applying the matrix
    before black level subtraction would transform the black level itself
    and require channel-specific offsets to undo.
    """
    h, w = rgb_f.shape[:2]
    # Maximum valid value after BLC (1.0 in normalized float = 65535 in 16-bit, minus BLC offset)
    max_val = 1.0 - blc_f
    
    for r in prange(h):
        for c in range(w):
            # Step 1: Apply Black Level Correction
            red = max(0.0, rgb_f[r, c, 0] - blc_f)
            grn = max(0.0, rgb_f[r, c, 1] - blc_f)
            blu = max(0.0, rgb_f[r, c, 2] - blc_f)
            
            # Step 2: Apply RGB Gain (compensate for Picamera2 channel imbalance)
            red = red * r_gain
            blu = blu * b_gain
            # Green gain is implicitly 1.0
            
            # Step 2b: Clip gained channels to valid range
            # This handles already-clipped inputs (sprocket holes, etc.)
            red = min(max_val, red)
            blu = min(max_val, blu)
            
            # Step 3: Apply Color Correction Matrix
            if apply_ccm:
                rgb_f[r, c, 0] = ccm[0,0]*red + ccm[0,1]*grn + ccm[0,2]*blu
                rgb_f[r, c, 1] = ccm[1,0]*red + ccm[1,1]*grn + ccm[1,2]*blu
                rgb_f[r, c, 2] = ccm[2,0]*red + ccm[2,1]*grn + ccm[2,2]*blu
            else:
                rgb_f[r, c, 0], rgb_f[r, c, 1], rgb_f[r, c, 2] = red, grn, blu

# ============================================================================
# Frame processing & command-line interface
# ============================================================================

class FrameError(Exception):
    """Raised when a frame cannot be read or is not the expected geometry."""


def load_raw_frame(in_p):
    """Load and validate a raw Bayer frame.

    Returns a 2D uint16 array of exactly FRAME_HEIGHT x FRAME_WIDTH.
    Raises FrameError if the file cannot be read or the geometry is wrong,
    since a mis-shaped frame would debayer into plausible-looking but
    incorrect output.
    """
    try:
        raw_u = tifffile.imread(in_p)
    except Exception as e:
        raise FrameError(f"could not read TIFF: {e}") from e

    raw_u = np.squeeze(raw_u)

    if raw_u.ndim != 2:
        raise FrameError(
            f"expected a single-channel 2D frame, got shape {raw_u.shape}")

    if raw_u.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        raise FrameError(
            f"expected {FRAME_HEIGHT}x{FRAME_WIDTH}, "
            f"got {raw_u.shape[0]}x{raw_u.shape[1]}")

    return raw_u


def resolve_gains(raw_u, r_gain, b_gain, use_dig_gains):
    """Decide the final red and blue gains for a frame.

    Embedded gains are used when requested; explicit command-line gains
    override them. Green is always 1.0. Returns (r_gain, b_gain).
    """
    final_r_gain = 1.0
    final_b_gain = 1.0

    if use_dig_gains:
        dig_gains = extract_digital_gains(raw_u)
        if dig_gains:
            final_r_gain, final_b_gain = dig_gains
            log(f"Using embedded digital gains: "
                f"R={final_r_gain:.3f}, B={final_b_gain:.3f}")
        else:
            log("Could not extract embedded gains, "
                "falling back to 1.0 / 1.0", level="WARN")

    if r_gain is not None:
        final_r_gain = r_gain
    if b_gain is not None:
        final_b_gain = b_gain

    for name, gain in (("Red", final_r_gain), ("Blue", final_b_gain)):
        if gain > 5.0:
            log(f"{name} gain is very high ({gain:.3f})", level="WARN")

    return final_r_gain, final_b_gain


def process_file(in_p, out_p, blc_int=DEFAULT_BLACK_LEVEL, ccm=False, temp=DEFAULT_CCM_TEMPERATURE,
                 stab=False, is_exr=False, r_gain=None, b_gain=None,
                 use_dig_gains=False, exr_full=False):
    """Develop one raw frame and write the result.

    Raises FrameError on unreadable or mis-shaped input, and OSError if the
    result cannot be written.
    """
    raw_u = load_raw_frame(in_p)
    meta = extract_metadata(raw_u)

    final_r_gain, final_b_gain = resolve_gains(
        raw_u, r_gain, b_gain, use_dig_gains)

    raw_f = raw_u.astype(np.float32) / 65535.0
    rgb_f = np.zeros((raw_f.shape[0], raw_f.shape[1], 3), dtype=np.float32)
    vng_debayer_gbrg_float(raw_f, rgb_f)

    blc_f = float(blc_int) / 65535.0
    ccm_m, ccm_temp = get_ccm_matrix(temp)
    if ccm:
        log(f"Using CCM for {ccm_temp}K (requested {temp}K)")
    apply_blc_ccm_float(rgb_f, blc_f, ccm, ccm_m, final_r_gain, final_b_gain)

    res = rgb_f
    if stab:
        if meta is None:
            log("No metadata available, skipping crop stabilisation",
                level="WARN")
        elif not meta[4]:
            log("No sprocket detected (sentinel value), skipping crop")
        else:
            l, r, t, b = calculate_crop_params(
                meta[0], meta[2], meta[3], meta[1], FRAME_HEIGHT)
            res = rgb_f[t:b, l:r, :]

    if is_exr:
        # PIZ compression is lossless. HALF rounds to 16-bit float and is
        # smaller; FLOAT keeps full 32-bit precision. Unlike TIFF, EXR
        # preserves the extended range the CCM can produce (negatives and
        # values above 1.0).
        precision = pyexr.FLOAT if exr_full else pyexr.HALF
        pyexr.write(out_p, res, precision=precision,
                    compression=pyexr.PIZ_COMPRESSION)
    else:
        # 16-bit integer TIFF clips anything outside [0, 1].
        final = np.clip(np.round(res * 65535.0), 0, 65535).astype(np.uint16)
        tifffile.imwrite(out_p, final, photometric='rgb')

    log(f"Processed: {Path(in_p).name} -> {Path(out_p).name}")


def build_parser():
    temps = ", ".join(str(t) for t in CCM_TEMPS)
    parser = argparse.ArgumentParser(
        description="Float-precision debayering for GBRG raw frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Input must be {FRAME_HEIGHT}x{FRAME_WIDTH} single-channel "
               f"16-bit TIFF.\nIn batch mode the first error stops the run.")

    parser.add_argument('input', help='Input TIFF file, or directory with --batch')
    parser.add_argument('output', help='Output file, or directory with --batch')
    parser.add_argument('--batch', action='store_true',
                        help='Treat input and output as directories and '
                             'process every TIFF found')

    parser.add_argument('--blc', type=int, default=DEFAULT_BLACK_LEVEL,
                        metavar='LEVEL',
                        help=f'Black level subtracted from every channel, in '
                             f'16-bit counts (default: {DEFAULT_BLACK_LEVEL}). '
                             f'This runs whether or not --ccm is given; pass '
                             f'--blc 0 to disable it entirely.')

    parser.add_argument('--ccm', action='store_true',
                        help='Apply the colour correction matrix. This gates '
                             'only the matrix step, not black level correction.')
    parser.add_argument('--color-temp', type=int,
                        default=DEFAULT_CCM_TEMPERATURE, metavar='KELVIN',
                        help=f'Colour temperature for the CCM '
                             f'(default: {DEFAULT_CCM_TEMPERATURE}). '
                             f'The nearest calibrated matrix is used, with no '
                             f'interpolation. Calibrated at: {temps}.')

    parser.add_argument('--r-gain', type=float, default=None, metavar='GAIN',
                        help='Red channel gain, overriding any embedded gain')
    parser.add_argument('--b-gain', type=float, default=None, metavar='GAIN',
                        help='Blue channel gain, overriding any embedded gain')
    parser.add_argument('--use-dig-gains', action='store_true',
                        help='Read per-channel digital gains embedded in the '
                             'raw frame')

    parser.add_argument('--crop-stabilize', action='store_true',
                        help='Crop the frame to the detected sprocket hole. '
                             'Skipped with a warning if no sprocket was found.')

    parser.add_argument('--exr', action='store_true',
                        help='Write OpenEXR 16-bit half-float instead of TIFF. '
                             'Preserves values outside [0, 1] that TIFF clips.')
    parser.add_argument('--exr-full', action='store_true',
                        help='Write OpenEXR 32-bit float. Implies --exr.')

    return parser


def validate_args(args, parser):
    """Reject argument combinations that cannot produce correct output."""
    if args.blc < 0:
        parser.error("--blc must not be negative")
    if args.blc > 65535:
        parser.error("--blc must not exceed 65535")

    for name, gain in (("--r-gain", args.r_gain), ("--b-gain", args.b_gain)):
        if gain is not None and gain <= 0.0:
            parser.error(f"{name} must be positive")

    if (args.exr or args.exr_full) and pyexr is None:
        parser.error("EXR output requires pyexr; install it with "
                     "'pip install pyexr' or drop --exr / --exr-full")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    in_path, out_path = Path(args.input), Path(args.output)
    want_exr = args.exr or args.exr_full
    ext = ".exr" if want_exr else ".tif"

    common = dict(blc_int=args.blc, ccm=args.ccm, temp=args.color_temp,
                  stab=args.crop_stabilize, is_exr=want_exr,
                  r_gain=args.r_gain, b_gain=args.b_gain,
                  use_dig_gains=args.use_dig_gains, exr_full=args.exr_full)

    if args.batch:
        if not in_path.is_dir():
            log(f"{in_path} is not a directory", level="ERR")
            return 1

        files = sorted(f for f in in_path.iterdir()
                       if f.is_file() and f.suffix.lower() in ('.tif', '.tiff'))
        if not files:
            log(f"No TIFF files found in {in_path}", level="ERR")
            return 1

        try:
            out_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            log(f"Could not create {out_path}: {e}", level="ERR")
            return 1

        log(f"Processing {len(files)} files to {ext}")
        for i, f in enumerate(files):
            try:
                process_file(str(f), str(out_path / f.with_suffix(ext).name),
                             **common)
            except (FrameError, OSError) as e:
                log(f"Failed on {f.name}: {e}", level="ERR")
                log(f"Aborting: {i} of {len(files)} files completed.",
                    level="ERR")
                return 1

        log(f"Done: {len(files)} files processed.")
        return 0

    if not in_path.is_file():
        log(f"{in_path} is not a file", level="ERR")
        return 1

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        process_file(str(in_path), str(out_path), **common)
    except (FrameError, OSError) as e:
        log(f"Failed on {in_path.name}: {e}", level="ERR")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
