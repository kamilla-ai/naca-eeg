import torch
import numpy as np
import os

def get(mini=False, fixed_order=False):
    data = {}
    taskcla = []
    inputsize = [8, 1, 3000]  # (channels, height, width)
    labelsize = 5             # количество классов сна (если у тебя другое — поменяй)

    # Путь к данным — замени, если нужно
    root_dir = "C:/Users/Kamilla/Downloads/ISRUC"
    subject_list = sorted([d for d in os.listdir(root_dir) if d.isdigit()])

    if not fixed_order:
        np.random.shuffle(subject_list)

    if mini:
        subject_list = subject_list[:5]

    for t, subject in enumerate(subject_list):
        subject_path = os.path.join(root_dir, subject)
        data_dir = os.path.join(subject_path, "data")
        label_dir = os.path.join(subject_path, "label")

        data_files = sorted(os.listdir(data_dir))
        label_files = sorted(os.listdir(label_dir))

        xs, ys = [], []
        for fdata, flabel in zip(data_files, label_files):
            x = np.load(os.path.join(data_dir, fdata))     # shape: (N, 8, 3000)
            y = np.load(os.path.join(label_dir, flabel))   # shape: (N,)
            # Если метки — строки, конвертируем их в числа
            if y.dtype.type is np.str_ or y.dtype.type is np.object_:
                try:
                    y = y.astype(np.int64)
                except ValueError:
                    print("Unknown label strings in", flabel)
                    print("Example values:", np.unique(y))
                    raise

            # x: (N, 8, 3000) → (N, 8, 1, 3000) to match Conv input
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(2)
            y = torch.tensor(y, dtype=torch.long)

            xs.append(x)
            ys.append(y)

        xs = torch.cat(xs, dim=0)  # [N, 8, 1, 3000]
        ys = torch.cat(ys, dim=0)  # [N]

        n = len(xs)
        train_x, train_y = xs[:int(0.6 * n)], ys[:int(0.6 * n)]
        valid_x, valid_y = xs[int(0.6 * n):int(0.8 * n)], ys[int(0.6 * n):int(0.8 * n)]
        test_x,  test_y  = xs[int(0.8 * n):], ys[int(0.8 * n):]

        data[t] = {
            'name': f'subject-{subject}',
            'ncla': labelsize,
            'train': {'x': train_x, 'y': train_y},
            'valid': {'x': valid_x, 'y': valid_y},
            'test':  {'x': test_x,  'y': test_y},
        }

        taskcla.append((t, labelsize))

    data['ncla'] = labelsize
    return data, taskcla, inputsize, labelsize
