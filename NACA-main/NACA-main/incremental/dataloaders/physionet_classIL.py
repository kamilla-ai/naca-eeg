# dataloaders/physionet_classIL.py

import torch
import numpy as np
import os

def get(mini=False, fixed_order=False):
    data = {}
    taskcla = []
    size = [1, 64, 640]  # adjust to your saved EEG trial shape (C, H, W)
    labelsize = 4        # 4-class motor imagery

    root_dir = os.path.join(os.path.dirname(__file__), "..", "..", "BCI2000-4")
    root_dir = os.path.abspath(root_dir)
    subject_list = sorted(os.listdir(root_dir))

    if not fixed_order:
        np.random.shuffle(subject_list)

    for t, subject in enumerate(subject_list):
        subject_path = os.path.join(root_dir, subject)
        data_files = sorted(os.listdir(os.path.join(subject_path, "data")))
        label_files = sorted(os.listdir(os.path.join(subject_path, "label")))

        xs, ys = [], []
        for fdata, flabel in zip(data_files, label_files):
            x = np.load(os.path.join(subject_path, "data", fdata))  # shape: [64, 640]
            y = np.load(os.path.join(subject_path, "label", flabel))  # scalar (0–3)

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, 64, 640]
            y = torch.tensor(y, dtype=torch.long)

            xs.append(x)
            ys.append(y)

            if mini and len(xs) >= 100:
                break

        xs = torch.stack(xs)  # [N, 1, 64, 640]
        ys = torch.stack(ys)  # [N]

        data[t] = {
            'name': f'subject-{subject}',
            'ncla': labelsize,
            'train': {'x': xs[:int(0.6 * len(xs))], 'y': ys[:int(0.6 * len(ys))]},
            'valid': {'x': xs[int(0.6 * len(xs)):int(0.8 * len(xs))], 'y': ys[int(0.6 * len(ys)):int(0.8 * len(ys))]},
            'test':  {'x': xs[int(0.8 * len(xs)):], 'y': ys[int(0.8 * len(ys)):]},
        }

        taskcla.append((t, labelsize))

    data['ncla'] = labelsize * len(subject_list)
    return data, taskcla, size, labelsize
