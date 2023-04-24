def printHello():
	print("TEST")                       
import RPi.GPIO as GPIO
import time
import spidev
#from lib_nrf24 import NRF24

def sendMessage():
	
	#change pins config from pin # to internal pin use
	GPIO.setmode(GPIO.BCM)

	#create permanent addres for the pipe
	pipes = [[0xe7,0xe7,0xe7,0xe7,0xe7],[0xc2,0xc2,0xc2,0xc2,0xc2]]

	#define radio
	radio=NRF24(GPIO, spidev.SpiDev())

	#Intialize the NRF module with the CE and CSN pins
	radio.begin(0,25) 
	#set payload size to 32 bits
	radio.setPayloadSize(32)
	radio.setChannel(0x76)
	radio.setDataRate(NRF24.BR_2MBPS)
	radio.setPALevel(NRF24.PA_MIN)
	radio.setAutoAck(True)
	radio.enableDynamicPayloads()
	radio.enableAckPayload()
	radio.openWritingPipe(pipes[1])
	radio.printDetails()
	while True:
		message=list("Person Detected")

		radio.write(message)
		print("We sent the message of{}".format(message))
		#check if returned an ackPL
		if radio.isAckPayloadAvailable():
			returnedPL=[]
			radio.read(returnedPL,radio.getDynamicPayloadSize())
			print("Return payload is {}".format(returnedPL))
		else:
			print("No Payload Recieved")
		time.sleep(1)
