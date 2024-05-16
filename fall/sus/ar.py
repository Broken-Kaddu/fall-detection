import serial.tools.list_ports
import serial


ports = serial.tools.list_ports.comports()

serial_inst = serial.Serial()


port_var=f'COM3'
ports_list = []



for port in ports:
    if str(port).startswith(port_var):
        break
    else:

        ports_list.append(str(port))

        print(str(port))



        val: str = input('Select Port: COM')



        for i in range(len(ports_list)):

         if ports_list[i].startswith(f'COM{val}'):

                port_var = f'COM{val}'

                print(port_var)



serial_inst.baudrate = 9600

serial_inst.port = port_var

serial_inst.open()




    

while True:
    command: str = input('Arduino Command: (ON/OFF): ').upper()
    print(command)
    serial_inst.write(command.encode('utf-8'))



    if command == 'EXIT':

        exit(0)