import numpy as np
import torch


def interaug(batch):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    current = 0
    n_chunks = 9 if new_samples.shape[-1] == 1125 else (8 if new_samples.shape[-1] % 8 == 0 else 7) # special case for BCIC III
    for cls in torch.unique(y):
        x_cls = x[y == cls]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        # indices = np.random.choice(len(x_cls), size=(len(x_cls), n_chunks),
        #                            replace=True)
        indices = torch.randint(0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device)  

        for idx in indices:
            # add offset
            idx += np.arange(0, chunks.shape[0], len(x_cls))

            # create new sample
            new_sample = chunks[idx]
            new_sample = new_sample.permute(1, 0, 2).reshape(
                1, x_cls.shape[1], x_cls.shape[2])
            new_samples[current] = new_sample
            new_labels[current] = cls
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    return combined_x, combined_y
