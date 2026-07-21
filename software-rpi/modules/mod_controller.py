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
Controller — scan sequencing and motion orchestration.

The central state machine driving the scanner. Services GUI button and
sequence requests, manages serial connect/disconnect and stepper
enable/disable handshakes, issues film-advance move commands, and runs the
capture -> save -> move scan loop including sprocket-marker realignment and
end-of-roll (tension) stop handling.

Shared memory:
    Reads:  Flags — SHUTDOWN_REQUESTED, SERIAL_CONNECTED,
            SERIAL_CONNECT_REQUEST, SERIAL_CONNECT_FAILED,
            SERIAL_DISCONNECT_REQUEST, READY_SERIAL_CONNECT,
            READY_SERIAL_DISCONNECT, STEPPERS_ENABLED, CMD_ENABLE_STEPPERS,
            CMD_DISABLE_STEPPERS, READY_ENABLE_STEPPERS,
            READY_DISABLE_STEPPERS, BUTTON_PRESSED, CMD_STEP_FWD/REV,
            CMD_FRAME_FWD/REV, CMD_X_FRAMES_FWD/REV, CMD_MOVE_TO_MARKER,
            CMD_SEQ_START, CMD_SEQ_STOP, TO_MARKER_FWD, READY_MOVE_CMD,
            READY_CAPTURE, SAVE_COMPLETE_CONTROLLER, MOVE_TENSION_LOW;
            Uint16 — FILM_LOCATION_VERTICAL, STEPS_TO_MARKER, STEPS_PER_FRAME,
            FRAME_COUNTER
    Writes: Flags — SERIAL_CONNECT_REQUEST, SERIAL_DISCONNECT_REQUEST,
            RQST_SERIAL_CONNECT, RQST_SERIAL_DISCONNECT, SERIAL_CONNECTED,
            SERIAL_CONNECT_FAILED, READY_SERIAL_CONNECT,
            READY_SERIAL_DISCONNECT, RQST_ENABLE_STEPPERS,
            RQST_DISABLE_STEPPERS, READY_ENABLE_STEPPERS,
            READY_DISABLE_STEPPERS, STEPPERS_ENABLED, BUTTON_DONE,
            MOVE_CMD_DIRECTION, RQST_MOVE_CMD, READY_MOVE_CMD, RQST_CAPTURE,
            SAVE_REQUEST, SAVE_COMPLETE_CONTROLLER, CMD_SEQ_START,
            CMD_SEQ_STOP, CMD_SEQ_ACTIVE, RQST_LED_OFF, and clears the CMD_*
            GUI request flags; Uint16 — CMD_SERIAL_ARGUMENT, FRAME_COUNTER

Interacts with:
    mod_serial      (serial connect/disconnect, move, and stepper
                    enable/disable handshakes; MOVE_CMD_DIRECTION,
                    CMD_SERIAL_ARGUMENT, MOVE_TENSION_LOW, RQST_LED_OFF)
    mod_raw_capture (RQST_CAPTURE/READY_CAPTURE; consumes FILM_LOCATION_VERTICAL,
                    STEPS_TO_MARKER, TO_MARKER_FWD produced by the detector)
    mod_file_saver  (SAVE_REQUEST/SAVE_COMPLETE_CONTROLLER)
    GUI panels      (BUTTON_PRESSED/BUTTON_DONE, CMD_* button and sequence
                    flags, serial and stepper enable requests)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, Uint16Index
from tpx_logger import log, LogTag
from tpx_config import DISABLE_STEPPERS_ON_DISCONNECT

import time


class ControllerProcessor:
    def __init__(self):
        log(LogTag.CTRL, "Initializing...")

        self.flags = get_array("flags")
        self.sm_uint16var = get_array("shm_uint16_var")

        if (self.flags is None or self.sm_uint16var is None):
            raise RuntimeError("Controller Module: Failed to get shared arrays")

        log(LogTag.CTRL, "Connected to shared memory")

        self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
        self.to_marker_steps = self.sm_uint16var[Uint16Index.STEPS_TO_MARKER]
        self.to_marker_direction = self.flags[FlagIndex.TO_MARKER_FWD]

        self.button_pressed = False
        self.cmd_step_fwd = False
        self.cmd_step_rev = False
        self.cmd_frame_fwd = False
        self.cmd_frame_rev = False
        self.cmd_x_frames_fwd = False
        self.cmd_x_frames_rev = False

        self.cmd_seq_start = False
        self.cmd_seq_stop = False
        self.cmd_seq_active = False
        self.ready_capture = False

        self.steps_per_frame = self.sm_uint16var[Uint16Index.STEPS_PER_FRAME]
        self.ready_serial = False

    def increment_frame_counter(self, delta):
        """Adjust the frame counter by delta, clamped to the valid uint16
        range so it never goes negative or overflows."""
        current = self.sm_uint16var[Uint16Index.FRAME_COUNTER]
        new_value = int(current) + delta

        if new_value < 0:
            new_value = 0

        if new_value > 65535:
            new_value = 65535

        self.sm_uint16var[Uint16Index.FRAME_COUNTER] = new_value

    def gui_move_flags_read(self):
        """Snapshot the GUI movement command flags into local state for this
        loop iteration."""
        self.button_pressed = self.flags[FlagIndex.BUTTON_PRESSED]
        self.cmd_step_fwd = self.flags[FlagIndex.CMD_STEP_FWD]
        self.cmd_step_rev = self.flags[FlagIndex.CMD_STEP_REV]
        self.cmd_frame_fwd = self.flags[FlagIndex.CMD_FRAME_FWD]
        self.cmd_frame_rev = self.flags[FlagIndex.CMD_FRAME_REV]
        self.cmd_x_frames_fwd = self.flags[FlagIndex.CMD_X_FRAMES_FWD]
        self.cmd_x_frames_rev = self.flags[FlagIndex.CMD_X_FRAMES_REV]
        self.cmd_mov_to_marker = self.flags[FlagIndex.CMD_MOVE_TO_MARKER]
        self.cmd_enable_stepper = self.flags[FlagIndex.CMD_ENABLE_STEPPERS]

    def gui_seq_flags_read(self):
        """Snapshot the GUI sequence start/stop flags into local state."""
        self.cmd_seq_start = self.flags[FlagIndex.CMD_SEQ_START]
        self.cmd_seq_stop = self.flags[FlagIndex.CMD_SEQ_STOP]

    def gui_flags_done(self):
        """Clear the GUI command flags after handling them and signal the GUI
        that the request is complete."""
        self.flags[FlagIndex.BUTTON_PRESSED] = False
        self.flags[FlagIndex.CMD_STEP_FWD] = False
        self.flags[FlagIndex.CMD_STEP_REV] = False
        self.flags[FlagIndex.CMD_FRAME_FWD] = False
        self.flags[FlagIndex.CMD_FRAME_REV] = False
        self.flags[FlagIndex.CMD_X_FRAMES_FWD] = False
        self.flags[FlagIndex.CMD_X_FRAMES_REV] = False
        self.flags[FlagIndex.CMD_MOVE_TO_MARKER] = False
        self.flags[FlagIndex.BUTTON_DONE] = True

    def cmd_mov_rqst_wait_ready(self):
        """Request a move from the Serial module and block until it reports the
        move complete via READY_MOVE_CMD."""
        self.flags[FlagIndex.READY_MOVE_CMD] = False
        self.flags[FlagIndex.RQST_MOVE_CMD] = True
        self.ready_serial = False

        while not self.ready_serial:
            time.sleep(0.1)
            self.ready_serial = self.flags[FlagIndex.READY_MOVE_CMD]

    def process(self):
        """Main service loop: run the serial-connection and stepper state
        machine, dispatch GUI move commands, and drive the capture -> save ->
        move scan sequence until shutdown is requested."""
        log(LogTag.CTRL, "Starting processing...")

        try:
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                # Handle serial connection state
                serial_connected = self.flags[FlagIndex.SERIAL_CONNECTED]

                if not serial_connected:
                    if self.flags[FlagIndex.SERIAL_CONNECT_REQUEST]:
                        self.flags[FlagIndex.SERIAL_CONNECT_REQUEST] = False

                        log(LogTag.CTRL, "Serial connect requested")

                        self.flags[FlagIndex.RQST_SERIAL_CONNECT] = True

                        ready_connect = False
                        timeout_counter = 0
                        while not ready_connect and timeout_counter < 100:
                            time.sleep(0.1)
                            ready_connect = self.flags[FlagIndex.READY_SERIAL_CONNECT]
                            timeout_counter += 1

                        if not ready_connect:
                            log(LogTag.CTRL, "Serial connection timeout", level="WARN")
                            self.flags[FlagIndex.SERIAL_CONNECT_FAILED] = True
                        else:
                            if not self.flags[FlagIndex.SERIAL_CONNECT_FAILED]:
                                self.flags[FlagIndex.SERIAL_CONNECTED] = True
                                log(LogTag.CTRL, "Serial connected successfully")
                            else:
                                log(LogTag.CTRL, "Serial connection failed", level="WARN")

                        self.flags[FlagIndex.READY_SERIAL_CONNECT] = False
                    else:
                        time.sleep(0.05)

                # Serial is connected — check for a disconnection request
                if self.flags[FlagIndex.SERIAL_DISCONNECT_REQUEST]:
                    self.flags[FlagIndex.SERIAL_DISCONNECT_REQUEST] = False

                    log(LogTag.CTRL, "Serial disconnect requested")

                    # Optional safety: disable steppers before disconnecting.
                    # Gated by DISABLE_STEPPERS_ON_DISCONNECT in tpx_config.py.
                    if DISABLE_STEPPERS_ON_DISCONNECT and self.flags[FlagIndex.STEPPERS_ENABLED]:
                        log(LogTag.CTRL, "Auto-disabling steppers before disconnect")
                        self.flags[FlagIndex.RQST_DISABLE_STEPPERS] = True

                        ready_disable = False
                        timeout_counter = 0
                        while not ready_disable and timeout_counter < 50:  # 5 second timeout
                            time.sleep(0.1)
                            ready_disable = self.flags[FlagIndex.READY_DISABLE_STEPPERS]
                            timeout_counter += 1

                        self.flags[FlagIndex.READY_DISABLE_STEPPERS] = False
                        self.flags[FlagIndex.STEPPERS_ENABLED] = False

                    self.flags[FlagIndex.RQST_SERIAL_DISCONNECT] = True

                    ready_disconnect = False
                    timeout_counter = 0
                    while not ready_disconnect and timeout_counter < 50:  # 5 second timeout
                        time.sleep(0.1)
                        ready_disconnect = self.flags[FlagIndex.READY_SERIAL_DISCONNECT]
                        timeout_counter += 1

                    self.flags[FlagIndex.READY_SERIAL_DISCONNECT] = False
                    self.flags[FlagIndex.SERIAL_CONNECTED] = False
                    log(LogTag.CTRL, "Serial disconnected")

                    continue  # Loop back to disconnected state

                # If steppers are enabled
                if self.flags[FlagIndex.STEPPERS_ENABLED]:

                    if self.flags[FlagIndex.CMD_DISABLE_STEPPERS]:
                        self.flags[FlagIndex.CMD_DISABLE_STEPPERS] = False

                        log(LogTag.CTRL, "Steppers disable requested")

                        self.flags[FlagIndex.RQST_DISABLE_STEPPERS] = True

                        self.ready_disable_stepper = False
                        while not self.ready_disable_stepper:
                            time.sleep(0.1)
                            self.ready_disable_stepper = self.flags[FlagIndex.READY_DISABLE_STEPPERS]

                        self.flags[FlagIndex.READY_DISABLE_STEPPERS] = False
                        self.flags[FlagIndex.STEPPERS_ENABLED] = False

                    self.gui_move_flags_read()

                    if self.button_pressed:
                        self.steps_per_frame = self.sm_uint16var[Uint16Index.STEPS_PER_FRAME]

                        # Step buttons
                        if self.cmd_step_fwd or self.cmd_step_rev:
                            self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = self.steps_per_frame // 20

                            if self.cmd_step_fwd:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = True # Forward
                                delta=0
                            else:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = False # Reverse
                                delta=0
                            self.cmd_mov_rqst_wait_ready()

                        # Frame buttons
                        if self.cmd_frame_fwd or self.cmd_frame_rev:
                            self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = self.steps_per_frame

                            if self.cmd_frame_fwd:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = True # Forward
                                delta=1
                            else:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = False # Reverse
                                delta=-1
                            self.cmd_mov_rqst_wait_ready()
                            self.increment_frame_counter(delta)

                        # X-frames buttons
                        if self.cmd_x_frames_fwd or self.cmd_x_frames_rev:
                            self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = self.steps_per_frame * 10

                            if self.cmd_x_frames_fwd:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = True # Forward
                                delta=10
                            else:
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = False # Reverse
                                delta=-10
                            self.cmd_mov_rqst_wait_ready()
                            self.increment_frame_counter(delta)

                        if self.cmd_mov_to_marker:
                            self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]

                            if self.film_vertical < 0xFFFF:
                                self.to_marker_steps = self.sm_uint16var[Uint16Index.STEPS_TO_MARKER]
                                self.to_marker_direction = self.flags[FlagIndex.TO_MARKER_FWD]
                                self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = self.to_marker_steps
                                self.flags[FlagIndex.MOVE_CMD_DIRECTION] = self.to_marker_direction
                                self.cmd_mov_rqst_wait_ready()
                                log(LogTag.CTRL, f"Move to marker — {self.to_marker_steps} steps")
                        self.gui_flags_done()
                    time.sleep(0.1)

                    self.gui_seq_flags_read()

                    if self.cmd_seq_start:
                        self.flags[FlagIndex.CMD_SEQ_START] = False
                        self.flags[FlagIndex.CMD_SEQ_ACTIVE] = True
                        self.cmd_seq_active = True

                        # Capture loop
                        while self.cmd_seq_active:

                            # Capture
                            self.flags[FlagIndex.RQST_CAPTURE] = True
                            self.ready_capture = False

                            while not self.ready_capture:
                                time.sleep(0.1)
                                self.ready_capture = self.flags[FlagIndex.READY_CAPTURE]

                            # Save (runs concurrently with the move below; confirmed later)
                            self.flags[FlagIndex.SAVE_COMPLETE_CONTROLLER] = False
                            self.flags[FlagIndex.SAVE_REQUEST] = True
                            save_ready = False

                            # Move
                            self.steps_per_frame = self.sm_uint16var[Uint16Index.STEPS_PER_FRAME]
                            self.flags[FlagIndex.MOVE_CMD_DIRECTION] = True # Forward
                            self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = int(self.steps_per_frame * 1)
                            self.cmd_mov_rqst_wait_ready()
                            self.increment_frame_counter(1)

                            time.sleep(0.6)

                            counter_out = 5
                            seq_fault = False

                            while counter_out > 0:
                                self.film_vertical = self.sm_uint16var[Uint16Index.FILM_LOCATION_VERTICAL]
                                if self.film_vertical < 0xFFFF : # TODO Confirm that a sprocket is detected in range
                                    self.to_marker_steps = self.sm_uint16var[Uint16Index.STEPS_TO_MARKER]
                                    self.to_marker_direction = self.flags[FlagIndex.TO_MARKER_FWD]
                                    if self.to_marker_steps < 400: #Limit error to 400 steps to avoid false detection
                                        self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT] = self.to_marker_steps
                                        self.flags[FlagIndex.MOVE_CMD_DIRECTION] = self.to_marker_direction
                                        self.cmd_mov_rqst_wait_ready()
                                        time.sleep(0.4)

                                        self.to_marker_steps = self.sm_uint16var[Uint16Index.STEPS_TO_MARKER]
                                    else:
                                        self.flags[FlagIndex.CMD_SEQ_STOP] = True
                                        seq_fault = True
                                        log(LogTag.CTRL, f"Sequence stopped — sprocket out of range ({self.to_marker_steps} steps)", level="WARN")
                                        break

                                    if self.to_marker_steps > 0x0010:
                                        counter_out = counter_out-1
                                    else:
                                        break
                                else:
                                    break

                            # Confirm the save completed before continuing
                            while not save_ready:
                                time.sleep(0.1)
                                save_ready = self.flags[FlagIndex.SAVE_COMPLETE_CONTROLLER]

                            # Check whether the loop should stop
                            # TODO NEEDS TO INCLUDE TENSION AS STOP
                            self.cmd_seq_stop = self.flags[FlagIndex.CMD_SEQ_STOP]
                            self.cmd_tension_low = self.flags[FlagIndex.MOVE_TENSION_LOW]

                            # If the stop is due to low tension, turn the LED off
                            if self.cmd_seq_stop or self.cmd_tension_low:
                                if self.cmd_tension_low:
                                    self.flags[FlagIndex.RQST_LED_OFF] = True
                                    log(LogTag.CTRL, "Sequence stopped — tension low", level="WARN")
                                elif seq_fault:
                                    pass  # already logged at fault point
                                else:
                                    log(LogTag.CTRL, "Sequence stopped — user requested")
                                self.flags[FlagIndex.CMD_SEQ_STOP] = False
                                self.flags[FlagIndex.CMD_SEQ_ACTIVE] = False
                                self.cmd_seq_active = False

                # If steppers are not enabled
                else:
                    if self.flags[FlagIndex.CMD_ENABLE_STEPPERS]:
                        self.flags[FlagIndex.CMD_ENABLE_STEPPERS] = False

                        log(LogTag.CTRL, "Steppers enable requested")

                        self.flags[FlagIndex.RQST_ENABLE_STEPPERS] = True

                        self.ready_stepper = False
                        while not self.ready_stepper:
                            time.sleep(0.1)
                            self.ready_stepper = self.flags[FlagIndex.READY_ENABLE_STEPPERS]

                        self.flags[FlagIndex.READY_ENABLE_STEPPERS] = False
                        self.flags[FlagIndex.STEPPERS_ENABLED] = True
                    else:
                        time.sleep(0.05)

        except KeyboardInterrupt:
            log(LogTag.CTRL, "Interrupted")
        except Exception as e:
            log(LogTag.CTRL, f"Error — {e}", level="ERR")


def main():
    """Process entry point: build the Controller and run its loop, releasing
    shared memory on exit."""
    try:
        processor = ControllerProcessor()
        processor.process()
    except Exception as e:
        log(LogTag.CTRL, f"Fatal error — {e}", level="ERR")
    finally:
        log(LogTag.CTRL, "Shutting down...")
        cleanup_manager()