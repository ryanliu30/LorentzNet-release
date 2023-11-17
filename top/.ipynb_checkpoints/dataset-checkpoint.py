from torch.utils.data import DataLoader
from . import collate_fn, initialize_datasets

def retrieve_dataloaders(batch_size, num_workers = 4, num_train = -1, datadir = './data'):
    # Initialize dataloader
    datasets = initialize_datasets(datadir, num_pts={'train':num_train,'test':-1,'valid':-1})
    # Construct PyTorch dataloaders from datasets
    collate = lambda data: collate_fn(data, scale=1, add_beams=True, beam_mass=1)
    dataloaders = {split: DataLoader(dataset,
                                     batch_size=batch_size,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     drop_last= True if (split == 'train') else False,
                                     num_workers=num_workers,
                                     collate_fn=collate)
                        for split, dataset in datasets.items()}

    return dataloaders