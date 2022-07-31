import rtmidi
from rtmidi.midiconstants import NOTE_ON, NOTE_OFF
import time

midiout = rtmidi.MidiOut()
midiout.open_virtual_port("My virtual output")

note_on = [NOTE_ON, 60, 132] # channel 1, middle C, velocity 112
note_off = [NOTE_OFF, 60, 0]
midiout.send_message(note_on)
time.sleep(1)
midiout.send_message(note_off)
time.sleep(0.1)

del midiout