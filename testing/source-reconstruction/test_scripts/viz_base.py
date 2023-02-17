import mne
from pathlib import Path

# sample_data_folder = mne.datasets.sample.data_path()
# subjects_dir = sample_data_folder / 'subjects'
subjects_dir = Path('~/data/freesurfer/sub-2')
Brain = mne.viz.get_brain_class()
brain = Brain('sample', hemi='lh', surf='pial',
              subjects_dir=subjects_dir, size=(800, 600))
brain.add_annotation('aparc.a2009s', borders=False)