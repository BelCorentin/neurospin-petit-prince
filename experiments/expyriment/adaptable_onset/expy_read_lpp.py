#! /usr/bin/env python3

""" Display text, word by word, at the center of the screen.

    Usage: 

     rsvp file.tsv

    where file.tsv is a tab-separated-values files with three colums: 
    word, onset, duration 
    (onset and duration are in seconds)
"""

import argparse
from queue import PriorityQueue
# import pandas as pd
import expyriment
from expyriment import stimuli
from expyriment.misc import Clock
import numpy as np
import textgrids   # pip install praat-textgrids
import math

TEXT_FONT = "Inconsolata.ttf"  
TEXT_SIZE = 40
TEXT_COLOR = (255, 255, 255)  # white
BACKGROUND_COLOR = (64, 64, 64)  # grey
WINDOW_SIZE = 1024, 768

# *****************************************
# Command-line arguments

parser = argparse.ArgumentParser()

# parser.add_argument('csv_files',
#                     nargs='+',
#                     action="append",
#                     default=[])
parser.add_argument("--text-font",
                    type=str,
                    default=TEXT_FONT,
                    help="set the font for text stimuli")
parser.add_argument("--text-size",
                    type=int,
                    default=TEXT_SIZE,
                    help="set the vertical size of text stimuli")
parser.add_argument("--text-color",
                    nargs='+',
                    type=int,
                    default=TEXT_COLOR,
                    help="set the font for text stimuli")
parser.add_argument("--background-color",
                    nargs='+',
                    type=int,
                    default=BACKGROUND_COLOR,
                    help="set the background color")
parser.add_argument("--window-size",
                    nargs='+',
                    type=int,
                    default=WINDOW_SIZE,
                    help="in window mode, sets the window size")
args = parser.parse_args()
TEXT_SIZE = args.text_size
TEXT_COLOR = tuple(args.text_color)
TEXT_FONT = args.text_font
BACKGROUND_COLOR = tuple(args.background_color)
WINDOW_SIZE = tuple(args.window_size)

# Open the text grid file
print('')
grid = textgrids.TextGrid("./annot.TextGrid")
grid_size = 1200  # To be redefined later

# stimlist = pd.read_csv(args.csv_files[0][0], 
# sep="\t", quoting=True, quotechar="*")


###############################
expyriment.control.defaults.window_mode = True
# expyriment.control.defaults.window_size = WINDOW_SIZE
# expyriment.design.defaults.experiment_background_colour = BACKGROUND_COLOR

exp = expyriment.design.Experiment(name="RSVP",
                                   background_colour=BACKGROUND_COLOR,
                                   foreground_colour=TEXT_COLOR,
                                   text_size=TEXT_SIZE,
                                   text_font=TEXT_FONT)
expyriment.control.initialize(exp)
exp._screen_colour = BACKGROUND_COLOR
kb = expyriment.io.Keyboard()


####################################################
# Prepare the queue of events 
bs = stimuli.BlankScreen(colour=BACKGROUND_COLOR)
events = PriorityQueue()
map_text_surface = dict()


# Initial function working with the way the tsv file is built
# Printing a word every 300ms to the screen

# for row in stimlist.itertuples():
#     text = row.word
#     onset = row.onset 
#     duration = row.duration

# Secondary function, for the textgrids files
# Printing every word for its duration in the audio file

# The purpose of the offset is to track everytime
# We add some delay to the current words
# Eg: the word "de" only has a duration of 60ms, to which
# We add 200ms to make it visible, we then have to shift 
# All of the next words
offset = 0

for i in np.arange(grid_size):

    text = grid["text words"][i].text
    next_text = grid["text words"][i+1].text

    # There are some empty textual intervals in the textgrid
    # Ignore them
    if text == "":
        continue

    # Adapt the duration if the next word interval is empty
    if next_text == "":
        duration = float(grid["text words"][i + 2].xmin) - \
            float(grid["text words"][i].xmin)
        next_onset = float(grid["text words"][i + 2].xmin)
        # print(f"Empty word after {text}")

    else:
        duration = float(grid["text words"][i + 1].xmin) - \
            float(grid["text words"][i].xmin)
        next_onset = float(grid["text words"][i + 1].xmin)

    onset = float(grid["text words"][i].xmin)

    if text in map_text_surface.keys():
        stim = map_text_surface[text]
    else:
        stim = stimuli.TextLine(text,
                                text_font=TEXT_FONT,
                                text_size=TEXT_SIZE,
                                text_colour=TEXT_COLOR,
                                background_colour=BACKGROUND_COLOR)
        map_text_surface[text] = stim

    # # Modulate the duration of the text by its length
    # len_word = len(text)
    # modulation = (math.sqrt(len_word) * 200) + 1000
    # timing = onset * modulation
    # print(timing)
    # events.put((timing+offset, text, stim))

    # Try with a higher timing for each word
    # timing = 1200 * onset
    # events.put((timing+offset, text, stim))

    # Try with a minimum threshold for word length
    # duration = next_onset - onset
    # timing = 1000 * onset
    # offset_to_add = 0
    # if duration < 0.3:
    #     offset_to_add = (0.3 - duration)*1000
    #     print(f'Added {(0.3 - duration)*1000} to the offset for the word {text}')

    # events.put((timing+offset, text, stim))
    # print(f'Whats sent is timing : {timing}, offset: {offset, text}   \n')
    # offset += offset_to_add

    # Threshold mini & maxi
    duration = next_onset - onset
    timing = 1000 * onset
    offset_to_add = 0
    if duration < 0.3:
        offset_to_add = (0.3 - duration)*1000
        print(f'Added {(0.3 - duration)*1000} to the offset for the word {text}')

    if duration > 0.6:
        offset_to_add = (0.6 - duration)*1000
        # print(f'Added {(0.3 - duration)*1000} to the offset for the word {text}')

    events.put((timing+offset, text, stim))
    print(f'Whats sent is timing : {timing}, offset: {offset, text}   \n')
    offset += offset_to_add

    

    
    # Not needed anymore
    # events.put(((onset + duration) * 1000, "", bs))


#############################################################
# let's go

# for i in np.arange(100):
#     print(events.get())
expyriment.control.start(subject_id=0)

a = Clock()

previous_onset = 0

while not events.empty():
    onset, text, stim = events.get()

    while a.time < (onset - 10):
        a.wait(1)
        k = kb.check()
        if k is not None:
            exp.data.add([a.time, 'keypressed,{}'.format(k)])

    stim.present()
    print(f"Duration = {onset - previous_onset}\n")
    previous_onset = onset

