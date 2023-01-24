#! /usr/bin/env python3
# Time-stamp: <2023-01-07 18:06:25 chistophe@pallier.org>

""" Display text, word by word, at the center of the screen.

    Usage: 

     rsvp --chapter 4

    where 4 is the number of the chapter wanted
"""

import argparse
from queue import PriorityQueue
import pandas as pd
import expyriment
from expyriment import stimuli, io
from expyriment.misc import Clock

# expyriment.control.set_develop_mode(on=True,
#                                     intensive_logging=False,
#                                     skip_wait_methods=True)

expyriment.control.defaults.window_mode = False


# VERSION CHOICE:
# Version 1: 350 ms between each word; 300 ms of word + 50 ms of black screen
# Version 2: 250 ms / 50 ms + 500 ms at the end of each sentence
VERSION = 2

# Triggers
port_address_output = "/dev/parport1"
port1 = io.ParallelPort(port_address_output)

# Const
TEXT_FONT = "Inconsolata.ttf"
TEXT_SIZE = 40
# TEXT_COLOR = (255, 255, 255)  # white
TEXT_COLOR = (230, 230, 230)  # white but not too white
# BACKGROUND_COLOR = (64, 64, 64)  # grey
BACKGROUND_COLOR = (30, 30, 30)  # black
WINDOW_SIZE = 1024, 768
CHAPTER = 1
FIXED_WORD_DURATION = 200  # Overriding tsv file
FIXED_BS_DURATION = 50  # Overriding tsv file
SPEED = 0.9
END_OF_SENTENCE_BLANK = True

######## command-line arguments
parser = argparse.ArgumentParser()

# parser.add_argument('csv_files',
#                     nargs='+',
#                     action="append",
#                     default=[])
parser.add_argument(
    "--text-font", type=str, default=TEXT_FONT, help="set the font for text stimuli"
)
parser.add_argument(
    "--text-size",
    type=int,
    default=TEXT_SIZE,
    help="set the vertical size of text stimuli",
)
parser.add_argument(
    "--text-color",
    nargs="+",
    type=int,
    default=TEXT_COLOR,
    help="set the font for text stimuli",
)
parser.add_argument(
    "--background-color",
    nargs="+",
    type=int,
    default=BACKGROUND_COLOR,
    help="set the background color",
)
parser.add_argument(
    "--window-size",
    nargs="+",
    type=int,
    default=WINDOW_SIZE,
    help="in window mode, sets the window size",
)
parser.add_argument(
    "--chapter",
    nargs="+",
    type=int,
    default=CHAPTER,
    help="choose which chapter to play",
)
args = parser.parse_args()
TEXT_SIZE = args.text_size
TEXT_COLOR = tuple(args.text_color)
TEXT_FONT = args.text_font
BACKGROUND_COLOR = tuple(args.background_color)
WINDOW_SIZE = tuple(args.window_size)
CHAPTER = args.chapter[0]

if VERSION == 1:
    csv_file = f"./../formatting/v1/run{CHAPTER}_v1_word_0.3_end_sentence_0.2.tsv"
else:
    csv_file = f"./../formatting/v2/run{CHAPTER}run1_v1_word_0.3_end_sentence_0.2.tsv"
# stimlist = pd.read_csv(args.csv_files[0][0], sep="\t", quoting=True, quotechar="*")
stimlist = pd.read_csv(csv_file, sep="\t", quoting=True, quotechar="*")

###############################
# expyriment.control.defaults.window_mode = True
# expyriment.control.defaults.window_size = WINDOW_SIZE
# expyriment.design.defaults.experiment_background_colour = BACKGROUND_COLOR

exp = expyriment.design.Experiment(
    name="RSVP",
    background_colour=BACKGROUND_COLOR,
    foreground_colour=TEXT_COLOR,
    text_size=TEXT_SIZE,
    text_font=TEXT_FONT,
)
expyriment.control.initialize(exp)
exp._screen_colour = BACKGROUND_COLOR
kb = expyriment.io.Keyboard()


####################################################
# Prepare the queue of events
bs = stimuli.BlankScreen(colour=BACKGROUND_COLOR)
photodiode = stimuli.Rectangle((90, 90), position=(900, -500))


events = PriorityQueue()
map_text_surface = dict()

for row in stimlist.itertuples():
    text = row.word

    onset = row.onset
    duration = row.duration

    if text in map_text_surface.keys():
        stim = map_text_surface[text]
    else:
        stim = stimuli.TextLine(
            text,
            text_font=TEXT_FONT,
            text_size=TEXT_SIZE,
            text_colour=TEXT_COLOR,
            background_colour=BACKGROUND_COLOR,
        )
        map_text_surface[text] = stim

    events.put((onset * 1000 * SPEED, text, stim))
    events.put(((onset + duration) * 1000 * SPEED, "", bs))


#############################################################
# let's go
expyriment.control.start(subject_id=0)

# init triggers
port1.send(data=0)


a = Clock()

# Initialize

port1.send(data=CHAPTER)

while not events.empty():
    onset, text, stim = events.get()
    value_trigger = len(text)
    while a.time < (onset - 10):
        a.wait(1)
        k = kb.check()
        if k is not None:
            exp.data.add([a.time, "keypressed,{}".format(k)])
    port1.send(data=value_trigger)
    stim.present()
    if value_trigger == 0:
        photodiode.present()
