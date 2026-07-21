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
Shared memory index definitions.

Single source of truth for all shared-memory slot indices: coordination flags
(FlagIndex), sprocket-position layout (SprocketIndex), and the typed scalar
variable arrays (FloatIndex, Uint16Index, Uint32Index, StringIndex). Imported
by every module that reads or writes shared memory; defines indices only.

Shared memory:
    Reads:  -
    Writes: -

Interacts with:
    -
"""

class FlagIndex:
    """Shared memory flag indices for inter-module coordination"""

    #Compile cache sequence
    MOD_READY_RAWCAPTURE = 1
    MOD_READY_DEBAYER = 2
    MOD_READY_SPROCKET = 3
    MOD_READY_DETECTOR = 4
    MOD_READY_WAVEFORM = 5
    MOD_READY_FILESAVER = 6
    MOD_READY_GUI_AOI = 7
    MOD_READY_ALL = 8 #Completion of all modules

    #Image pipeline
    RQST_DBY_B0 = 9 # Request Debayering of B0
    RQST_DBY_B1 = 10 # Request Debayering of B1
    RQST_SPK_B0 = 11 # Request Sprocket Detection of B0
    RQST_SPK_B1 = 12 # Request Sprocket Detection of B1
    RQST_DBY_RGB0 = 13 # Request Debayering of RGB0
    RQST_DBY_RGB1 = 14 # Request Debayering of RGB1
    RQST_WVF_0 = 15 # Request Creation of Waveform 0
    RQST_WVF_1 = 16 # Request Creation of Waveform 1
    READY_WVF0 = 17 # Waveform 0 Ready
    READY_WVF1 = 18 # Waveform 1 Ready
    READY_RGB0 = 19 # RGB0 Ready
    READY_RGB1 = 20 # RGB1 Ready
    READY_AOI0 = 21 # AOI0 Ready
    READY_AOI1 = 22 # AOI1 Ready

    #LED Direct Serial Changes
    RQST_LED_OFF = 23
    RQST_LED_DET = 24
    RQST_LED_RGB = 25
    RQST_LED_R = 26
    RQST_LED_G = 27
    RQST_LED_B = 28
    READY_LED_OFF = 29
    READY_LED_DET = 30
    READY_LED_RGB = 31
    READY_LED_R = 32
    READY_LED_G = 33
    READY_LED_B = 34
    RQST_LED_TAKE = 35  # Live Take L
    READY_LED_TAKE = 36  # Live Take L

    #Capture Flags
    RQST_CAPTURE = 37 # Request Capture
    READY_CAPTURE = 38 # Ready Capture

    #GUI Buttons
    BUTTON_PRESSED = 39 # A Scan Button was Pressed
    BUTTON_DONE = 40 # Controller Done with Button Task
    CMD_STEP_FWD = 41
    CMD_STEP_REV = 42
    CMD_FRAME_FWD = 43
    CMD_FRAME_REV = 44
    CMD_X_FRAMES_FWD = 45
    CMD_X_FRAMES_REV = 46
    CMD_MOVE_TO_MARKER = 47

    #Controller – Serial Flags
    RQST_MOVE_CMD = 48 # Request Send Serial Command
    READY_MOVE_CMD = 49 # Serial Command Sent
    MOVE_CMD_DIRECTION = 50 # Direction of Move

    CMD_SEQ_START = 51 # Start Scanning Sequence
    CMD_SEQ_STOP = 52 # Stop Scanning Sequence
    CMD_SEQ_ACTIVE = 53 # Sequence Active

    MOVE_TENSION_LOW = 54 # Flags when either Pickup or Supply Tension is Low

    STEPPERS_ENABLED = 55 # Status: True when steppers enabled
    SERIAL_CONNECTED = 56 # Status: True when serial port connected
    TRANSPORT_CONTROLS_LOCKED = 57 # Computed: True when controls should be grayed

    CMD_ENABLE_STEPPERS = 58 # GUI requests stepper enable
    CMD_DISABLE_STEPPERS = 59 # GUI requests stepper disable
    SERIAL_CONNECT_REQUEST = 60 # GUI requests serial connection
    SERIAL_DISCONNECT_REQUEST = 61 # GUI requests serial disconnection

    STEPPERS_ENABLE_FAILED = 62 # Set when enable command fails
    SERIAL_CONNECT_FAILED = 63 # Set when connect/reconnect fails

    RQST_ENABLE_STEPPERS = 64 # Controller → Serial
    READY_ENABLE_STEPPERS = 65 # Serial → Controller
    RQST_DISABLE_STEPPERS = 66 # Controller → Serial
    READY_DISABLE_STEPPERS = 67 # Serial → Controller

    # Serial connection flags
    RQST_SERIAL_CONNECT = 68    # Controller → Serial
    READY_SERIAL_CONNECT = 69   # Serial → Controller
    RQST_SERIAL_DISCONNECT = 70 # Controller → Serial
    READY_SERIAL_DISCONNECT = 71 # Serial → Controller

    # Capture Flags
    RQST_RAW_CAPTURE_RGB_WRITE = 72 # raw_buffer_16 to raw_capture_16
    READY_RAW_CAPTURE_RGB_WRITE = 73 # raw_buffer_16 to raw_capture_16 ready
    LOCK_BUFFER_16_WRITE = 74 # Halt writes to buffer_16
    READY_BUFFER_16 = 75 # On Buffer 16 write

    #TODO Verfiy if any of the following FlagIndex are stale
    LOCK_CAPTURE_16_READ = 76
    MODE_CAPTURE_RGB = 77

    # File saver flags
    SAVE_REQUEST = 78  # GUI/Controller requests save
    SAVE_COMPLETE = 79  # Save operation completed
    SAVE_COMPLETE_CONTROLLER = 80
    SAVE_RAW = 81  # Save as raw (vs debayered)

    #File Saver
    SAVE_APPLY_CROP = 82

    TO_MARKER_FWD = 83

    # Capture Module
    CLK_CAPTURE = 84

    RAW_FRAME_CLK = 85
    RGB_8_SHOW_BUFFER = 86

    RGB_8_FOCUS = 87
    RGB_8_FOCUS_INDICATOR = 88
    WVF_PARADE = 89
    DBY_HALT = 90
    WVF_HALT = 91
    SPROCKET_HALT = 92

    RGB_8_COLOR_PROCESSING = 93  # Apply BLC + Gain + CCM to Raw Monitor and Waveform
    RGB_8_INVERT = 94  # Invert Raw Monitor display (negative preview); waveform follows

    # GUI module flags
    GUI_READY = 95
    GUI_LOCK_AOI_PANEL = 96  # Lock AOI panel controls during auto scanning
    SHUTDOWN_REQUESTED = 97

class SprocketIndex:

    # Index as Frame, Hole, Detected Value
    POS_FRAME0 = 0
    POS_FRAME1 = 1
    POS_HOLE0 = 0
    POS_HOLE1 = 1

    POS_TOP = 0
    POS_CENTER = 1
    POS_BOTTOM = 2
    POS_EDGE = 3

class FloatIndex:

    FT_CAPTURE_TIME = 0

    # Digital gain for R and B channels (float32, range 1.0 - 3.0)
    # Written by GUI gain panel, read by mod_debayer and mod_file_saver
    R_DIG_GAIN = 1
    B_DIG_GAIN = 2

class Uint16Index:
    SPROCKET_MARKER = 1 #This is 16 bit
    CROP_HEIGHT = 2
    CAPTURE_DEPTH_R = 3 #This is 8 bit
    CAPTURE_DEPTH_G = 4 #This is 8 bit
    CAPTURE_DEPTH_B = 5 #This is 8 bit
    STACKING_SETTING = 6 #This is 8 bit

    #scancontrol_panel
    LED_OFF = 7
    LED_DET = 8
    LED_RGB = 9
    LED_R = 10
    LED_G = 11
    LED_B = 12

    #AOI TO BE CHANGED BY FILM SELECTION
    AOI_SPROCKET_LOCATION = 13
    AOI_POSITION = 14

    FILM_LOCATION_VERTICAL = 15
    FILM_LOCATION_HORIZONTAL = 16

    STEPS_PER_FRAME = 17
    PIX_TO_STEPS = 18
    STEPS_TO_MARKER = 19

    FRAME_COUNTER = 20 #TEMPORARY consider using Uint32

    CMD_SERIAL = 21
    CMD_SERIAL_ARGUMENT = 22
    LED_TAKE = 23  # Live light value for Take L (separate from LED_RGB capture setting)

class Uint32Index:
    FRAME_COUNTER = 0 #Increments with every frame
    BUFFER_COUNTER = 1  #Increments with every buffer update
    EXPOSURE = 2
    RAW0_EXPOSURE = 3
    RAW1_EXPOSURE = 4

    #scancontrol_panel
    EXP_DET = 6
    EXP_RGB = 7
    EXP_R = 8
    EXP_G = 9
    EXP_B = 10

class StringIndex:
    """Shared memory string array indices"""

    # Define what each string slot is used for
    STATUS_MESSAGE = 0
    ERROR_MESSAGE = 1
    CAMERA_SETTINGS = 2
    FILE_PATH = 3
    USER_COMMENT = 4
    TIMESTAMP = 5
    METADATA_1 = 6
    METADATA_2 = 7
    DEBUG_INFO = 8
    SPARE = 9

    # String limits
    MAX_STRINGS = 10
    MAX_LENGTH = 260