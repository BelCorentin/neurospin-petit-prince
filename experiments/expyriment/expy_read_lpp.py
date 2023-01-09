import expyriment

exp = expyriment.design.Experiment(name="First Experiment")
expyriment.control.initialize(exp)

stim = expyriment.stimuli.TextLine(text="Word 1")
stim.preload()

expyriment.control.start()

stim.present()
exp.clock.wait(1000)

stim = expyriment.stimuli.TextLine(text="Word 2")
stim.preload()
stim.present()
exp.clock.wait(1000)

for i in np.range(1000):
    stim = expyriment.stimuli.TextLine(text=f"{i}")
    stim.preload()

    stim.present()
    exp.clock.wait(200)

expyriment.control.end()
