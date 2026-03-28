from torch.utils.data import DataLoader
from .shha import SHHA

def build_loader(config, mode):
    data_path = config.INPUT_DIR
    batch_size = config.BATCH_SIZE
    num_workers = config.NUM_WORKERS
    Dataset = {'sha': SHHA, 'shb': SHHA}[config.TYPE_DATASET.lower()]
    data_set = Dataset(data_path, mode)
    return DataLoader(data_set, batch_size=batch_size if (mode=='train') else 1, num_workers=num_workers, pin_memory=config.PIN_MEMORY, shuffle=(mode=='train'), collate_fn=Dataset.collate_fn)