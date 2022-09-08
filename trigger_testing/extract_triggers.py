import mne 

from mne_bids import read_raw_bids,print_dir_tree,make_report,BIDSPath
import matplotlib.pyplot as plt

bids_root = '/home/is153802/workspace_LPP/data/MEG/LPP/bids_dataset'


# print_dir_tree(bids_root)

# print(make_report(bids_root))

datatype = 'meg'
session = '01'
run = '1'
task = 'None'
suffix = 'meg'
subject = '190715'
bids_path = BIDSPath(root=bids_root, subject=subject,suffix = suffix, task=task,run=run, session=session, datatype=datatype)

print(bids_path.match())
print(bids_path)
bids_path
# 
raw = read_raw_bids(bids_path=bids_path, verbose=False)

raw.plot()

raw.plot_psd(average = True)

plt.show()