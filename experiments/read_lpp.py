from psychopy import visual, core

# Import textgrid

# Open the text grid file


win = visual.Window()
msg = visual.TextStim(win, text="\u00A1Hola mundo!")

msg.draw()
win.flip()
core.wait(1)
win.close()
