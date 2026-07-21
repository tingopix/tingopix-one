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
Debayer with black-level correction and color-correction matrix

Numba-JIT debayer library for full-resolution scanner output. Provides the
embedded per-temperature CCM table, a nearest-temperature CCM lookup, and the
stride-2 debayer that applies BLC + optional per-channel gain + CCM to produce
16-bit RGB. Pure compute — no shared memory or process state; imported by the
debayer modules.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    mod_debayer (get_ccm_matrix),
"""

import numpy as np
from numba import njit

# CCM matrices extracted from imx477_scientific_reference.json (project root),
# kept as the provenance record for these values. Note this is a copy of
# libcamera's imx477_scientific.json — the renamed file is documentation only
# and is never loaded; mod_capture.py loads the system tuning by its original
# name.
# 
# Available color temperatures (Kelvin)
CCM_TEMPS = np.array([
    2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
    4100, 4600, 5100, 5600, 6100, 6600, 7100, 7600, 8100, 8600
], dtype=np.int32)


# CCM matrices for each temperature (19 x 3 x 3)
CCM_MATRICES = np.array([
    # 2000K
    [[1.5813882365848004, -0.35293683714581114, -0.27378771561617715],
     [-0.4347297185453639, 1.5792631087746074, -0.12102601986382337],
     [0.2322290578987574, -1.4382672640468128, 2.1386425781770755]],
    # 2200K
    [[1.6322048484088305, -0.45932286857238486, -0.21373542690252198],
     [-0.3970719209901105, 1.5877868651467202, -0.17249380832122455],
     [0.20753774825903412, -1.2660673594740142, 2.005654261091916]],
    # 2400K
    [[1.6766610071470398, -0.5447101051688111, -0.16838641107407676],
     [-0.3659845183388154, 1.592223692670396, -0.2127091997471162],
     [0.1833964516767549, -1.1339155942419321, 1.9089342978542396]],
    # 2600K
    [[1.7161984340622154, -0.6152585785678794, -0.1331100845092582],
     [-0.33972082628066275, 1.5944888273736966, -0.2453979465898787],
     [0.1615577497676328, -1.0298684958833109, 1.8357854177422053]],
    # 2800K
    [[1.7519307259815728, -0.6748682080165339, -0.10515169074540848],
     [-0.3171703484479931, 1.5955820297498486, -0.2727395854813966],
     [0.14230870739974305, -0.9460976023551511, 1.778709391659538]],
    # 3000K
    [[1.7846716625128374, -0.7261240476375332, -0.08274697420358428],
     [-0.2975654035173307, 1.5960425637021738, -0.2961043416505157],
     [0.12546426281675097, -0.8773434727076518, 1.7330356805246685]],
    # 3200K
    [[1.8150085872943436, -0.7708109672515514, -0.06469468211419174],
     [-0.2803468940646277, 1.596168842967451, -0.3164044170681625],
     [0.11071494533513807, -0.8199772290209191, 1.69572135046367]],
    # 3400K
    [[1.8433668304932087, -0.8102060605062592, -0.05013485852801454],
     [-0.2650934036324084, 1.5961288492969294, -0.33427554893845535],
     [0.0977478941863518, -0.7714303112098978, 1.6647070820146963]],
    # 3600K
    [[1.8700575831917468, -0.8452518300291346, -0.03842644337477299],
     [-0.2514794528347016, 1.5960178299141876, -0.3501774949366156],
     [0.08628520830733245, -0.729841503339915, 1.638553343939267]],
    # 4100K
    [[1.8988700903560716, -0.8911278803351247, -0.018848644425650693],
     [-0.21487101487384094, 1.599236541382614, -0.39405450457918206],
     [0.08251488056482173, -0.7178919368326191, 1.6267009056502704]],
    # 4600K
    [[1.960355191764125, -0.9624344812121991, -0.0017122408632169205],
     [-0.19444620905212898, 1.5978493736948447, -0.416727638296156],
     [0.06310261513271084, -0.6483790952487849, 1.5834605477213093]],
    # 5100K
    [[2.014680536961399, -1.0195930302148566, 0.007728256612638915],
     [-0.17751999660735496, 1.5977081555831, -0.4366085498741474],
     [0.04741267583041334, -0.5950327902073489, 1.5512919847321853]],
    # 5600K
    [[2.062652337917251, -1.0658386679125478, 0.011886354256281267],
     [-0.16319197721451495, 1.598363237584736, -0.45422061523742235],
     [0.03465810928795378, -0.5535454108047286, 1.5269025836946852]],
    # 6100K
    [[2.104985902038069, -1.103597868736314, 0.012503517136539277],
     [-0.15090797064906178, 1.5994703078166095, -0.4698414300864995],
     [0.02421766063474242, -0.5208922818196823, 1.5081270847783788]],
    # 6600K
    [[2.1424988751299714, -1.134760232367728, 0.010730356010435522],
     [-0.14021846798466234, 1.600822462230719, -0.48379204794526487],
     [0.015521315410496622, -0.49463630325832275, 1.4933313534840327]],
    # 7100K
    [[2.1758034100130925, -1.1607558481037359, 0.007452724895469076],
     [-0.13085694672641826, 1.6022648614493245, -0.4962330524084075],
     [0.008226943206113427, -0.4733077192319791, 1.4815336120437468]],
    # 7600K
    [[2.205529206931895, -1.1826662383072108, 0.0032019529917605167],
     [-0.122572009780486, 1.6037258133595753, -0.5073973734282445],
     [0.0020132587619863425, -0.4556590236414181, 1.471939788496745]],
    # 8100K
    [[2.232224969223067, -1.2013672897252885, -0.0016234598095482985],
     [-0.11518026734442414, 1.6051544769439803, -0.5174558699422255],
     [-0.0033378143542219835, -0.4408590373867774, 1.4640252230667452]],
    # 8600K
    [[2.256082295891265, -1.2173210549996634, -0.0067231350481711675],
     [-0.10860272839843167, 1.6065150139140594, -0.5264728573611493],
     [-0.007952618707984149, -0.4284003574050791, 1.4574646927117558]]
], dtype=np.float32)


@njit
def get_ccm_matrix(color_temp):
    """
    Get CCM matrix for the given color temperature.
    If exact match not found, returns closest available temperature.
    
    Args:
        color_temp: Color temperature in Kelvin
        
    Returns:
        3x3 CCM matrix as float32 numpy array
    """
    # Find closest temperature
    min_diff = 999999
    best_idx = 0
    
    for i in range(len(CCM_TEMPS)):
        diff = abs(CCM_TEMPS[i] - color_temp)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    
    return CCM_MATRICES[best_idx]


@njit
def debayer_stride2_with_blc_ccm(raw_frame, bit_depth_r, bit_depth_g, bit_depth_b, rgb_output,
                                  black_r, black_g, black_b, color_temp, apply_ccm, apply_gain,
                                  gain_r, gain_b):
    """
    Debayer with stride-2 (every other pixel) to produce 1520x2032x3 RGB output.
    Applies Black Level Correction and Color Correction Matrix to debayered RGB.
    Handles bit depth conversion to always output 16-bit.
    
    GBRG Bayer pattern:
    G B G B ...
    R G R G ...
    
    Processing pipeline:
    1. Extract RGB from Bayer pattern (stride-2 sampling)
    2. Shift to 16-bit based on bit depth
    3. Apply Black Level Correction (per-channel)
    4. Apply Color Correction Matrix
    5. Clip to valid range
    
    Args:
        raw_frame: (3040, 4064) uint16 raw Bayer data
        bit_depth_r: Red channel bit depth (12-16)
        bit_depth_g: Green channel bit depth (12-16)
        bit_depth_b: Blue channel bit depth (12-16)
        rgb_output: (1520, 2032, 3) uint16 output array
        black_r: Black level for red channel, in 16-bit scale
        black_g: Black level for green channel, in 16-bit scale
        black_b: Black level for blue channel, in 16-bit scale
        color_temp: Color temperature in Kelvin for CCM selection
    """
    h_out, w_out = 1520, 2032
    
    # Calculate shift amounts for bit depth normalization
    shift_r = 16 - bit_depth_r
    shift_b = 16 - bit_depth_b
    shift_g_individual = 16 - bit_depth_g
    
    # Get CCM matrix for the specified color temperature
    ccm = get_ccm_matrix(color_temp)

    # Post-BLC saturation ceiling, shared by all three channels.
    # Derived from the largest black level so that every channel can actually
    # reach it; a smaller value would be unreachable for the channels carrying
    # a higher pedestal, leaving saturated pixels unbalanced across R, G, B.
    sat_level_f = np.float32(65535.0 - max(black_r, max(black_g, black_b)))

    # Pre-compute CCM-aware balanced clip level at saturation (computed once before pixel loop)
    # Run sat_level_f through each CCM row and use the minimum across all three channels
    r_sat_ccm = ccm[0, 0] * sat_level_f + ccm[0, 1] * sat_level_f + ccm[0, 2] * sat_level_f
    g_sat_ccm = ccm[1, 0] * sat_level_f + ccm[1, 1] * sat_level_f + ccm[1, 2] * sat_level_f
    b_sat_ccm = ccm[2, 0] * sat_level_f + ccm[2, 1] * sat_level_f + ccm[2, 2] * sat_level_f
    clip_ccm = min(r_sat_ccm, min(g_sat_ccm, b_sat_ccm))
    
    # Process each output pixel
    for i in range(h_out):
        for j in range(w_out):
            # Sample every other pixel (stride 2)
            row = i * 2
            col = j * 2
            
            # Extract 2x2 Bayer block
            # GBRG pattern:
            # [row+0, col+0] = G (top-left)
            # [row+0, col+1] = B
            # [row+1, col+0] = R
            # [row+1, col+1] = G (bottom-right)
            
            g1 = raw_frame[row, col]
            b_val = raw_frame[row, col + 1]
            r_val = raw_frame[row + 1, col]
            g2 = raw_frame[row + 1, col + 1]
            
            # Step 1 & 2: Apply shifts to normalize to 16-bit
            # R and B: simple left shift
            r_16 = np.uint16(r_val << shift_r)
            b_16 = np.uint16(b_val << shift_b)
            
            # Green: Shift each green first (to align to 15-bit), then add
            g1_shifted = np.uint16(g1 << shift_g_individual) >> 1
            g2_shifted = np.uint16(g2 << shift_g_individual) >> 1
            g_16 = np.uint16(g1_shifted + g2_shifted)
            
            # Step 3: Apply Black Level Correction
            # Convert to int32 to handle negative values during subtraction
            r_blc = np.int32(r_16) - black_r
            g_blc = np.int32(g_16) - black_g
            b_blc = np.int32(b_16) - black_b
            
            # Clip to positive range (BLC can result in negative values for dark pixels)
            r_blc = max(0, r_blc)
            g_blc = max(0, g_blc)
            b_blc = max(0, b_blc)
            
            # Step 4: Apply Color Correction Matrix
            # RGB_out = CCM @ RGB_in
            # Convert to float32 for CCM multiplication
            r_f = np.float32(r_blc)
            g_f = np.float32(g_blc)
            b_f = np.float32(b_blc)

            if apply_ccm :
                if apply_gain:
                    # Apply per-channel gain before CCM, clipping to sat_level_f
                    r_f = min(sat_level_f, r_f * gain_r)
                    g_f = min(sat_level_f, g_f)
                    b_f = min(sat_level_f, b_f * gain_b)

                # Matrix multiplication: [R', G', B'] = CCM × [R, G, B]
                r_ccm = ccm[0, 0] * r_f + ccm[0, 1] * g_f + ccm[0, 2] * b_f
                g_ccm = ccm[1, 0] * r_f + ccm[1, 1] * g_f + ccm[1, 2] * b_f
                b_ccm = ccm[2, 0] * r_f + ccm[2, 1] * g_f + ccm[2, 2] * b_f
                
                # Step 5: Clip to CCM-aware balanced ceiling and store
                rgb_output[i, j, 0] = np.uint16(min(clip_ccm, max(np.float32(0), r_ccm)))
                rgb_output[i, j, 1] = np.uint16(min(clip_ccm, max(np.float32(0), g_ccm)))
                rgb_output[i, j, 2] = np.uint16(min(clip_ccm, max(np.float32(0), b_ccm)))

            else:

                # Step 5: Clip to valid uint16 range and store
                rgb_output[i, j, 0] = np.uint16(min(65535, max(0, r_f)))
                rgb_output[i, j, 1] = np.uint16(min(65535, max(0, g_f)))
                rgb_output[i, j, 2] = np.uint16(min(65535, max(0, b_f)))