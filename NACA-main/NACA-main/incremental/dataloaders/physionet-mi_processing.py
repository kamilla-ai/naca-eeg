import os
import numpy as np
import mne
from mne.datasets import eegbci
import collections

# Output directory
save_root = "BCI2000-4"  # Change to absolute path if needed

# Subject list (1–109 in EEGBCI)
subjects = list(range(1, 101))
runs = [4, 6, 8, 10, 12, 14]  # motor imagery runs

if __name__ == '__main__':
    seconds = []
    consistent = []

    for idx, subject in enumerate(subjects):
        # Download EDFs
        raw_fnames = eegbci.load_data(subject, runs, path="datasets")

        # Create output folders
        subj_dir = os.path.join(save_root, f"{idx}")
        os.makedirs(os.path.join(subj_dir, "data"), exist_ok=True)
        os.makedirs(os.path.join(subj_dir, "label"), exist_ok=True)

        i = 0
        for fname in raw_fnames:
            run_code = fname.stem[-2:] # 'R04.edf' → '04'
            if run_code not in ['04', '06', '08', '10', '12', '14']:
                continue

            raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
            events, event_dict = mne.events_from_annotations(raw)
            raw_data = raw.get_data()

            for num in range(events.shape[0]):
                event_time = events[num, 0]
                event_code = events[num, 2]

                if event_code not in [2, 3]:
                    continue

                start = event_time
                end = start + 640

                if end <= raw_data.shape[1]:
                    window = raw_data[:, start:end]  # shape: [channels, time]

                    # Label assignment
                    if event_code == 2:
                        label = 0 if run_code in ['04', '08', '12'] else 2
                    else:
                        label = 1 if run_code in ['04', '08', '12'] else 3

                    np.save(os.path.join(subj_dir, "data", f"{i}.npy"), window)
                    np.save(os.path.join(subj_dir, "label", f"{i}.npy"), label)
                    i += 1

            if len(events) > 1:
                consistent.append(events[1, 0])
            seconds.append(raw_data.shape[1] // raw.info['sfreq'])

    print("Recording durations (in seconds):", collections.Counter(seconds))
    print("Event triggers [for checking]:", collections.Counter(consistent))
