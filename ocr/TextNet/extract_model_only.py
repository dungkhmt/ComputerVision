from torchtext.utils.iotools import save_checkpoint
import pickle
import torch
from functools import partial
import glob

list_model = glob.glob('./log/se_resnext101_32x4d-final-text-net-total-text-768-2/*.pth.tar')
print(list_model)
for p in list_model:
    if p == './log/se_resnext101_32x4d-final-text-net-total-text-no-randomcrop/quick_save_checkpoint_ep46.pth.tar':
        continue
    # if p == './log/se_resnext101_32x4d-final-text-net-total-text/quick_save_checkpoint_ep201.pth.tar':
    #     continue
    model_path = p

    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    checkpoint = torch.load(model_path, pickle_module=pickle)
    if 'optimizer' not in checkpoint.keys():
        continue
    pretrain_dict = checkpoint['state_dict']
    epoch = checkpoint['epoch']
    avg_loss = checkpoint['avg_loss']
    save_checkpoint({
                    'state_dict': pretrain_dict,
                    'epoch': epoch,
                    'avg_loss': avg_loss,
                    }, False, model_path)
