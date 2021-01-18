# Tactile-Music

This project is for Dhanya Nair's HEART lab of Chapman University.  The general idea is to take any arbitrary wav audio file on an external computer, do some signal processing, and send commands to an ESP32 microcontroller to actuate motors dependent on the output signals.

TestSignal.py is the WIP signal processing code.
BleakTest.py is a quick test code that verifies bleak's functionality and verifies the microcontroller BLE code.
main.py is the microcontroller code to receive commands and actuate motors dependent on those commands (or more specifically, the code outputs high or low on GPIO pins and external circuitry ensures that actuates a motor instead of flashing an LED or something).
