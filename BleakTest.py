#BleakTest.py is the code from a communicating computer.  Is some simplified code to test that main.py is correctly responding to communications.

from bleak import BleakClient
import bleak
import asyncio
import keyboard
import time

TARGET_UUID = 'b7328f9c-c89e-4d74-9a5e-000000000000'
UART_TX = 'b7328f9c-c89e-4d74-9a5e-000000000001' #UART'S TX is Bleak's RX
UART_RX = 'b7328f9c-c89e-4d74-9a5e-000000000002' #UART'S RX is Bleak's TX

async def read_test():
    TARGET_UUID = 'b7328f9c-c89e-4d74-9a5e-000000000000'
    UART_TX = 'b7328f9c-c89e-4d74-9a5e-000000000001' #UART'S TX is Bleak's RX
    UART_RX = 'b7328f9c-c89e-4d74-9a5e-000000000002' #UART'S RX is Bleak's TX

    address = "A4:CF:12:58:72:2A"  #NOTE: MAC address is per device, so this needs to be changed
    client = BleakClient(address)

    await client.connect()

    read_string = b''
    read_string = await client.read_gatt_char(UART_TX)
    print("TEST STRING: ", read_string.decode('UTF-8'))

    wDuty = 0
    aDuty = 0
    sDuty = 0
    dDuty = 0
    while True:
        if(keyboard.is_pressed('w')):
            wDuty = wDuty + 10
            if(wDuty > 1023):
                wDuty = 1023
            print("W is pressed!")
        else:
            wDuty = wDuty - 10
            if(wDuty < 0):
                wDuty = 0
        if(keyboard.is_pressed('a')):
            aDuty = aDuty + 10
            if(aDuty > 1023):
                aDuty = 1023
            print("A is pressed!")
        else:
            aDuty = aDuty - 10
            if(aDuty < 0):
                aDuty = 0
        if(keyboard.is_pressed('s')):
            sDuty = sDuty + 10
            if(sDuty > 1023):
                sDuty = 1023
            print("S is pressed!")
        else:
            sDuty = sDuty - 10
            if(sDuty < 0):
                sDuty = 0
        if(keyboard.is_pressed('d')):
            dDuty = dDuty + 10
            if(dDuty > 1023):
                dDuty = 1023
            print("D is pressed!")
        else:
            dDuty = dDuty - 10
            if(dDuty < 0):
                dDuty = 0
        if(keyboard.is_pressed('q')):
            print("Breaking")
            break
        time.sleep(0.025)

        #Sending all 4 motor control signals in a single packet compressed to bytes for efficiency purposes.
        write_bytes = wDuty.to_bytes(2, 'big') + aDuty.to_bytes(2, 'big') + sDuty.to_bytes(2, 'big') + dDuty.to_bytes(2, 'big')
        await client.write_gatt_char(UART_RX, write_bytes)


asyncio.run(read_test())
