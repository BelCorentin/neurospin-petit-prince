from psychopy import visual, core
import numpy as np
import textgrids # pip install praat-textgrids

# Open the text grid file
grid = textgrids.TextGrid("./annot.TextGrid")

win = visual.Window()
msg = visual.TextStim(win, text="Starting")

msg.draw()
win.flip()
core.wait(1)

for i in np.arange(100):
    word = grid["text words"][i].text

    time = int(grid["text words"][i + 1].xmin) - int(grid["text words"][i].xmin)
    msg = visual.TextStim(win, text=f" {word}")

    msg.draw()
    win.flip()
    core.wait(time)

win.close()
