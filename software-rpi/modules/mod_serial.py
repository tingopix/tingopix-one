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
Serial Processor — hardware-agnostic transport and light control.

Owns all communication with the scanner hardware. Selects a hardware
adapter at startup based on HARDWARE_MODE in tpx_config.py (TingopixAdapter
for the Pico/FCode default, CustomAdapter for third-party rigs), then runs a
flag-driven loop that services connect/disconnect, LED, film-advance, and
stepper-enable/disable requests from the Controller. All hardware access is
confined to the adapters; the loop itself never touches a serial port.

Shared memory:
    Reads:  RQST_SERIAL_CONNECT, RQST_SERIAL_DISCONNECT,
            RQST_LED_G/R/B/RGB/DET/OFF/TAKE, STEPPERS_ENABLED,
            RQST_MOVE_CMD, MOVE_CMD_DIRECTION, RQST_ENABLE_STEPPERS,
            RQST_DISABLE_STEPPERS, MOD_READY_ALL, SHUTDOWN_REQUESTED;
            Uint16: LED_G/R/B/RGB/DET/OFF/TAKE, CMD_SERIAL_ARGUMENT
    Writes: SERIAL_CONNECTED, SERIAL_CONNECT_FAILED, READY_SERIAL_CONNECT,
            READY_SERIAL_DISCONNECT, READY_LED_G/R/B/RGB/DET/OFF/TAKE,
            READY_MOVE_CMD, MOVE_TENSION_LOW, READY_ENABLE_STEPPERS,
            READY_DISABLE_STEPPERS

Interacts with:
    mod_controller (all requests/ready handshakes above via flags)
"""
from modules.memory.shm_manager import get_array, cleanup_manager
from modules.memory.shm_indexing import FlagIndex, Uint16Index
from tpx_logger import log, LogTag
from tpx_config import HARDWARE_MODE, SERIAL_PORT

import time
import serial


# HARDWARE MODE SELECTION
# Set HARDWARE_MODE in tpx_config.py:
#   "TINGOPIX" — TingoPix scanner with Pico controller (default)
#   "CUSTOM"   — Third-party scanner with user-provided hardware adapter


# TingoPix Adapter — Pico serial communication (default hardware)
# Do not modify this class unless changing TingoPix hardware behaviour.

class SerialComm:
    def __init__(self):
        self.cmd_reply = "None"
        self.cmd_tension_pickup = 0
        self.cmd_tension_supply = 0
        self.usb_serial_port = serial.Serial()
        self.tension_low = True

    def init_usbserial(self):
        log(LogTag.SER, f"Opening {SERIAL_PORT}...")
        self.usb_serial_port.port = SERIAL_PORT
        # Nominal only: the Pico CDC stack ignores the requested line rate, so
        # throughput is set by the USB bulk endpoints. Not user-configurable.
        self.usb_serial_port.baudrate = 115200
        self.usb_serial_port.bytesize = 8
        self.usb_serial_port.stopbits = 1
        self.usb_serial_port.parity = 'N'
        self.usb_serial_port.timeout= 1
        self.usb_serial_port.rtscts = 0
        self.usb_serial_port.open()

    def convert_tension(self, tension_data):
        tension_supply_str = tension_data[1:5]
        tension_pickup_str = tension_data[6:10]

        self.cmd_tension_supply = int(tension_supply_str, 16)
        self.cmd_tension_pickup = int(tension_pickup_str, 16)

        if self.cmd_tension_supply >= 0x8000:
            self.cmd_tension_supply = 0x0000 # Tension Negative, clip to zero
        if self.cmd_tension_pickup >= 0x8000:
            self.cmd_tension_pickup = 0x0000 # Tension Negative, clip to zero

        if self.cmd_tension_supply < 0x0020 or self.cmd_tension_pickup < 0x0020:
            self.tension_low = True
        else:
            self.tension_low = False

    def send_command(self, cmd_send, receive_timeout):
        """Send a command over the serial port, then read back the reply and
        update tension status. Skips the write if the port is not open."""
        if not self.usb_serial_port.is_open:
            log(LogTag.SER, f"Port not open — command not sent: [{cmd_send}]", level="WARN")
            return False

        try:
            cmd_uni = "["+str(cmd_send)+"]"
            cmd_ascii = cmd_uni.encode('raw_unicode_escape')
            self.usb_serial_port.timeout = 0.1
            self.usb_serial_port.write(cmd_ascii)

            self.usb_serial_port.timeout = receive_timeout
            self.cmd_reply = self.usb_serial_port.read_until(b'}')
            tension_data = self.cmd_reply[-12:-2]
            self.usb_serial_port.read()  # read CR
            self.usb_serial_port.read()  # read NL
            self.convert_tension(tension_data)
            return True
        except Exception as e:
            log(LogTag.SER, f"Command error — {e}", level="ERR")
            return False

    def send_steps(self, steps=0, fwd_direction=True):

        if fwd_direction:
            steps_ascii = f"X<{steps:04X}"
        else:
            steps_ascii = f"X>{steps:04X}"

        steps_timeout = (steps * 200e-6) + 0.5  # TODO this needs WORK.
        self.send_command(steps_ascii,steps_timeout)

    def send_light(self, light_setting=0, dac_channel=7, settling_time_ms = 500):
        light_setting_ascii = f"E{dac_channel:01x}{light_setting:03x}"
        self.send_command(light_setting_ascii, 0.3)
        time.sleep(settling_time_ms / 1000.0)

    def close_usbserial(self):
        log(LogTag.SER, "Closing serial port")
        self.usb_serial_port.close()


class TingopixAdapter:
    """
    Hardware adapter for TingoPix scanner with Pico controller.
    Wraps SerialComm — all film advance and light control goes through
    the Pico via USB serial (FCode protocol).
    """
    def __init__(self):
        self.com_port = SerialComm()
        self.tension_low = True

    def connect(self):
        """Open the serial port and perform the initial handshake with the
        Pico. Returns True once the Pico acknowledges, False otherwise."""
        log(LogTag.SER, f"Connecting to {SERIAL_PORT}...")
        self.com_port.init_usbserial()
        success = self.com_port.send_command("[]", 0.3)
        if not success:
            log(LogTag.SER, "Initial handshake failed", level="WARN")
            self.com_port.close_usbserial()
            return False
        log(LogTag.SER, "Connected successfully")
        return True

    def disconnect(self):
        """Turn the light off and close the serial port cleanly. Returns True
        on success, False if an error occurs while closing."""
        log(LogTag.SER, "Disconnecting...")
        try:
            if self.com_port.usb_serial_port.is_open:
                self.com_port.send_light(0x000, 0x7)
                time.sleep(0.1)
            self.com_port.close_usbserial()
            log(LogTag.SER, "Disconnected successfully")
            return True
        except Exception as e:
            log(LogTag.SER, f"Disconnect error — {e}", level="ERR")
            return False

    def enable_motors(self):
        """Enable all stepper axes on the Pico (capstan, supply, pickup) and
        home the transport."""
        log(LogTag.SER, "Steppers enabled")
        self.com_port.send_command("CE", 0.3)
        self.com_port.send_command("SE", 0.3)
        self.com_port.send_command("PE", 0.3)
        self.com_port.send_command("MP000", 0.3)

    def disable_motors(self):
        """Disable all stepper axes on the Pico so the transport can be moved
        by hand."""
        log(LogTag.SER, "Steppers disabled")
        self.com_port.send_command("CD", 0.3)
        self.com_port.send_command("SD", 0.3)
        self.com_port.send_command("PD", 0.3)

    def advance_film(self, steps, fwd_direction):
        """Advance the film by the requested number of steps, then refresh the
        cached tension status so the loop can signal end-of-roll."""
        self.com_port.send_steps(steps, fwd_direction)
        self.tension_low = self.com_port.tension_low

    def set_light(self, light_setting, led_idx):
        """
        Set the light level on the Pico DAC for the requested channel.
        led_idx: 0=G, 1=R, 2=B, 3=RGB, 4=DET, 5=OFF, 6=TAKE
        Capture channels (0-3) use 500ms settling because the light is
        changed mid-capture and the DAC must fully settle before the sensor
        integrates; DET, OFF, and TAKE are non-capture state changes and use
        the shorter 100ms settle.
        """
        if led_idx < 4:
            self.com_port.send_light(light_setting, 7, 500)
        else:
            self.com_port.send_light(light_setting, 7, 100)

    def shutdown(self):
        """Best-effort cleanup on exit: turn the light off and close the port
        if it is still open, tolerating a partially-initialized adapter."""
        try:
            if hasattr(self.com_port, 'usb_serial_port') and self.com_port.usb_serial_port.is_open:
                self.com_port.send_light(0x000, 0x7)
                log(LogTag.SER, "Light turned off")
                time.sleep(0.1)
                self.com_port.close_usbserial()
            else:
                log(LogTag.SER, "Port already closed or not initialized")
        except Exception as e:
            log(LogTag.SER, f"Error during shutdown — {e}", level="ERR")


# Custom Hardware Adapter
# This adapter is selected when HARDWARE_MODE = "CUSTOM" in tpx_config.py.
#
# Implement the five methods below to support your scanner hardware.
# The rest of the system (Controller, flags, shared memory) is unchanged.
#
# CONTRACT:
#   - connect()       must return True on success, False on failure
#   - disconnect()    must return True on success, False on failure
#   - enable_motors() is called once when the user enables transport
#   - disable_motors() is called once when the user disables transport
#   - advance_film()  is called for every frame advance — it may block
#                     until the move is complete; the Controller waits
#   - set_light()     is called before each capture channel — implement
#                     or leave as no-op; READY flag is always raised
#                     regardless so the system never blocks on light
#   - shutdown()      called on exit — safe to leave as no-op if not needed
#
# tension_low: set self.tension_low = True to signal end-of-roll to the
#              Controller (stops the scan sequence). If your hardware has
#              no tension sensing, leave it False permanently.

class CustomAdapter:
    def __init__(self):
        self.tension_low = False  # Set True to trigger end-of-roll stop

    def connect(self):
        # CUSTOM: Initialize your hardware here.
        # Examples: open a GPIO chip, open a different serial port,
        # configure a stepper driver, etc.
        # Return True if ready, False if initialization failed.
        log(LogTag.SER, "Custom adapter — connect() not implemented, skipping")
        return True

    def disconnect(self):
        # CUSTOM: Release your hardware here.
        # Examples: close GPIO, close serial port, disable driver power.
        # Return True on success, False on failure.
        log(LogTag.SER, "Custom adapter — disconnect() not implemented, skipping")
        return True

    def enable_motors(self):
        # CUSTOM: Enable your motor driver here.
        # Examples: assert an ENABLE pin, energize the coils,
        # send an enable command to an external controller.
        log(LogTag.SER, "Custom adapter — enable_motors() not implemented, skipping")

    def disable_motors(self):
        # CUSTOM: Disable your motor driver here.
        # Examples: de-assert ENABLE pin, de-energize coils.
        log(LogTag.SER, "Custom adapter — disable_motors() not implemented, skipping")

    def advance_film(self, steps, fwd_direction):
        # CUSTOM: Advance the film by the requested number of steps.
        # This method may block until the move is complete —
        # the Controller will wait at READY_MOVE_CMD.
        #
        # Arguments:
        #   steps         — number of steps to move (int)
        #   fwd_direction — True = forward, False = reverse (bool)
        #
        # Examples:
        #   GPIO pulse train to a stepper driver (A4988, DRV8825, etc.):
        #     set direction pin, then emit 'steps' pulses on step pin
        #
        #   Wait for an external trigger (frame sensor, limit switch):
        #     block here until the trigger input goes high
        #
        # End-of-roll detection:
        #   Set self.tension_low = True before returning if your hardware
        #   detects an end-of-roll condition (limit switch, torque sensor,
        #   frame counter, etc.). The process loop reads self.tension_low
        #   after every advance_film() call and signals the Controller,
        #   which will stop the scan sequence automatically — exactly as
        #   the TingoPix tension sensor does. If your hardware has no
        #   end-of-roll sensing, leave self.tension_low = False (default)
        #   and the sequence will run until manually stopped.
        log(LogTag.SER, f"Custom adapter — advance_film() not implemented (steps={steps} fwd={fwd_direction}), skipping")

    def set_light(self, light_setting, led_idx):
        # CUSTOM: Set the light level for the requested channel.
        # This is optional — leaving it as a no-op is fine.
        # The READY_LED flag is always raised by the process loop
        # regardless of what happens here, so the system never blocks.
        #
        # Arguments:
        #   light_setting — 12-bit intensity value 0x000–0xFFF (int)
        #   led_idx       — channel: 0=G, 1=R, 2=B, 3=RGB, 4=DET, 5=OFF, 6=TAKE
        #
        # Examples:
        #   PWM output on a GPIO pin scaled to light_setting
        #   I2C DAC write
        #   Serial command to an external light controller
        log(LogTag.SER, f"Custom adapter — set_light() not implemented (setting={light_setting:#05x} ch={led_idx}), skipping")

    def shutdown(self):
        # CUSTOM: Clean up on exit.
        # Examples: turn off lights, disable motors, close handles.
        # Safe to leave as a no-op if not needed.
        log(LogTag.SER, "Custom adapter — shutdown() not implemented, skipping")


# Serial Processor — process loop (hardware-agnostic)
# Selects adapter at startup based on HARDWARE_MODE in tpx_config.py.
# All flag handling and ready signalling is unchanged regardless of adapter.

class SerialProcessor:
    def __init__(self):
        log(LogTag.SER, "Initializing...")

        self.flags = get_array("flags")
        self.sm_uint16var = get_array("shm_uint16_var")

        self.steppers_enabled = False

        if (self.flags is None or self.sm_uint16var is None):
            raise RuntimeError("Serial Module: Failed to get shared arrays")

        log(LogTag.SER, "Connected to shared memory")

        # Select hardware adapter based on tpx_config.HARDWARE_MODE
        if HARDWARE_MODE == "CUSTOM":
            log(LogTag.SER, "Hardware mode: CUSTOM adapter selected")
            self.adapter = CustomAdapter()
        else:
            if HARDWARE_MODE != "TINGOPIX":
                log(LogTag.SER, f"Unknown HARDWARE_MODE '{HARDWARE_MODE}' — defaulting to TINGOPIX", level="WARN")
            log(LogTag.SER, "Hardware mode: TINGOPIX adapter selected")
            self.adapter = TingopixAdapter()

        self.flags[FlagIndex.SERIAL_CONNECTED] = False
        self.light_toggle = True

        log(LogTag.SER, "Initialized — waiting for connect request")

    def serial_connect(self):
        """Connect through the active adapter, catching any adapter error and
        reporting it as a failed connection."""
        try:
            return self.adapter.connect()
        except Exception as e:
            log(LogTag.SER, f"Connection failed — {e}", level="ERR")
            return False

    def serial_disconnect(self):
        """Disconnect through the active adapter, catching and logging any
        error raised while closing."""
        try:
            return self.adapter.disconnect()
        except Exception as e:
            log(LogTag.SER, f"Disconnect error — {e}", level="ERR")
            return False

    def serial_steppers_enable(self):
        self.adapter.enable_motors()

    def serial_steppers_disable(self):
        self.adapter.disable_motors()

    def shutdown(self):
        """Shut down through the active adapter, tolerating any error so
        cleanup can continue."""
        log(LogTag.SER, "Shutting down...")
        try:
            self.adapter.shutdown()
        except Exception as e:
            log(LogTag.SER, f"Error during shutdown — {e}", level="ERR")

    def process(self):
        """Main service loop: wait for all modules to be ready, then poll the
        request flags and dispatch connect, light, move, and stepper commands
        to the active adapter until shutdown is requested."""
        log(LogTag.SER, "Starting processing...")

        flag_index_led_request = [FlagIndex.RQST_LED_G, FlagIndex.RQST_LED_R, FlagIndex.RQST_LED_B, FlagIndex.RQST_LED_RGB, FlagIndex.RQST_LED_DET, FlagIndex.RQST_LED_OFF, FlagIndex.RQST_LED_TAKE]
        flag_index_led_ready = [FlagIndex.READY_LED_G, FlagIndex.READY_LED_R, FlagIndex.READY_LED_B, FlagIndex.READY_LED_RGB, FlagIndex.READY_LED_DET, FlagIndex.READY_LED_OFF, FlagIndex.READY_LED_TAKE]
        led_setting_index = [Uint16Index.LED_G, Uint16Index.LED_R, Uint16Index.LED_B, Uint16Index.LED_RGB, Uint16Index.LED_DET, Uint16Index.LED_OFF, Uint16Index.LED_TAKE]

        try:
            while not self.flags[FlagIndex.MOD_READY_ALL]:
                time.sleep(0.5)

            log(LogTag.SER, "Running")
            while not self.flags[FlagIndex.SHUTDOWN_REQUESTED]:

                # === Handle Serial Connection Requests ===

                if self.flags[FlagIndex.RQST_SERIAL_CONNECT]:
                    self.flags[FlagIndex.READY_SERIAL_CONNECT] = False
                    self.flags[FlagIndex.RQST_SERIAL_CONNECT] = False

                    success = self.serial_connect()

                    if not success:
                        self.flags[FlagIndex.SERIAL_CONNECT_FAILED] = True

                    self.flags[FlagIndex.READY_SERIAL_CONNECT] = True

                if self.flags[FlagIndex.RQST_SERIAL_DISCONNECT]:
                    self.flags[FlagIndex.READY_SERIAL_DISCONNECT] = False
                    self.flags[FlagIndex.RQST_SERIAL_DISCONNECT] = False

                    self.serial_disconnect()
                    self.flags[FlagIndex.READY_SERIAL_DISCONNECT] = True

                # === Handle Light Requests ===
                # READY flag is always raised after set_light() returns,
                # regardless of success — callers never block on light.

                for led_idx in range(7):
                    led_rqst_flag = self.flags[flag_index_led_request[led_idx]]
                    if led_rqst_flag:
                        led_setting = self.sm_uint16var[led_setting_index[led_idx]]
                        self.adapter.set_light(led_setting, led_idx)
                        self.flags[flag_index_led_request[led_idx]] = False
                        self.flags[flag_index_led_ready[led_idx]] = True

                # === Handle Move and Stepper Requests ===

                self.steppers_enabled = self.flags[FlagIndex.STEPPERS_ENABLED]
                if self.steppers_enabled:
                    # Handle move request
                    mov_rqst = self.flags[FlagIndex.RQST_MOVE_CMD]
                    if mov_rqst:
                        self.flags[FlagIndex.RQST_MOVE_CMD] = False
                        mov_dir = self.flags[FlagIndex.MOVE_CMD_DIRECTION]
                        mov_steps = self.sm_uint16var[Uint16Index.CMD_SERIAL_ARGUMENT]
                        self.adapter.advance_film(mov_steps, mov_dir)
                        self.flags[FlagIndex.RQST_MOVE_CMD] = False
                        self.flags[FlagIndex.READY_MOVE_CMD] = True
                        self.flags[FlagIndex.MOVE_TENSION_LOW] = self.adapter.tension_low

                    # Handle stepper disable request
                    rqst_disable_steppers = self.flags[FlagIndex.RQST_DISABLE_STEPPERS]
                    if rqst_disable_steppers:
                        self.flags[FlagIndex.READY_DISABLE_STEPPERS] = False
                        self.flags[FlagIndex.RQST_DISABLE_STEPPERS] = False
                        self.serial_steppers_disable()
                        self.flags[FlagIndex.READY_DISABLE_STEPPERS] = True

                else:
                    # Handle stepper enable request
                    rqst_enable_steppers = self.flags[FlagIndex.RQST_ENABLE_STEPPERS]
                    if rqst_enable_steppers:
                        self.flags[FlagIndex.READY_ENABLE_STEPPERS] = False
                        self.flags[FlagIndex.RQST_ENABLE_STEPPERS] = False
                        self.serial_steppers_enable()
                        self.flags[FlagIndex.READY_ENABLE_STEPPERS] = True
                    else:
                        time.sleep(0.001)

                time.sleep(0.05)

        except KeyboardInterrupt:
            log(LogTag.SER, "Interrupted by user")
        except Exception as e:
            log(LogTag.SER, f"Error — {e}", level="ERR")
        finally:
            self.shutdown()


def main():
    """Process entry point: build the Serial Processor, run its loop, and
    ensure the adapter is shut down and shared memory released on exit."""
    processor = None
    try:
        processor = SerialProcessor()
        processor.process()
    except Exception as e:
        log(LogTag.SER, f"Fatal error — {e}", level="ERR")
    finally:
        if processor is not None:
            processor.shutdown()
        cleanup_manager()
        log(LogTag.SER, "Cleanup complete")


if __name__ == "__main__":
    main()