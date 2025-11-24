from torch.utils.data import DataLoader
import argparse
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from visdom import Visdom
import time
import os
from tqdm import tqdm

from dataloader.plgcDataset import HsiRGBDataset
from utils.losses import BinaryDiceLoss
from utils.matrics import *
from utils.utils import set_seed, cal_seg_metric, cal_test_metrics, save_predict_map

from framework.model import RFFNet
import Compare.TransUnet.vit_configs as vit_configs

RGB_IN = 3
HSI_IN = 1
N_BAND = 16
N_CLASS = 2
VIT_NAME = 'R50-ViT-B_16'
VIT_IMG_SIZE =128
VIT_PATCH_SIZE = 16
VIT_N_SKIP = 2
VIT_CONFIGS = {
    'R50-ViT-B_16': vit_configs.get_r50_b16_config(),
    'R50-ViT-L_16': vit_configs.get_r50_l16_config()
}
config_vit = VIT_CONFIGS[VIT_NAME]
config_vit.n_classes = N_CLASS
config_vit.n_skip = VIT_N_SKIP
if VIT_NAME.find('R50') != -1:
    config_vit.patches.grid = (int(VIT_IMG_SIZE / VIT_PATCH_SIZE), int(VIT_IMG_SIZE / VIT_PATCH_SIZE))
criterion = {
        'dice_loss': BinaryDiceLoss()
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='results/res_PisaNet/plgc')
    parser.add_argument('--epochs', default=100, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N')

    # training
    parser.add_argument('--model', '-a', default='MyModel')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--checkpoint_path', default='results/res_PisaNet/plgc-model-v2/model.pkl')
    parser.add_argument('--lr', '--learning_rate', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--milestones', default='20,40,60', type=str)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--early_stopping_patience', default=10, type=int, metavar='N')

    # dataset
    parser.add_argument('--hsi_train', default=r'E:\Assignment\Datasets\PLGC\train\hyper\images')
    parser.add_argument('--hsi_val', default=r'E:\Assignment\Datasets\PLGC\val\hyper\images')
    parser.add_argument('--hsi_test', default=r'E:\Assignment\Datasets\PLGC\test\hyper\images')
    parser.add_argument('--mask_train', default=r'E:\Assignment\Datasets\PLGC\train\hyper\masks')
    parser.add_argument('--mask_val', default=r'E:\Assignment\Datasets\PLGC\val\hyper\masks')
    parser.add_argument('--mask_test', default=r'E:\Assignment\Datasets\PLGC\test\hyper\masks')
    parser.add_argument('--rgb_train', default=r'E:\Assignment\Datasets\PLGC\train\digi\images')
    parser.add_argument('--rgb_val', default=r'E:\Assignment\Datasets\PLGC\val\digi\images')
    parser.add_argument('--rgb_test', default=r'E:\Assignment\Datasets\PLGC\test\digi\images')
    parser.add_argument('--rgb_aligned', default=r'E:\Assignment\Datasets\PLGC\color_stitch')
    parser.add_argument('--txt_root', default=r'dataloader/plgc_txt')
    parser.add_argument('--prototype_root', default=r'dataloader/prototypes/PLGC_new_npy')

    config = parser.parse_args()
    return config


def main():
    window = Visdom()
    window.line([[0.12, 0.12]], [0.], win='Loss', opts=dict(title='Loss', legend=['train', 'val']))
    window.line([[0.77, 0.77]], [0.], win='Dice', opts=dict(title='Dice', legend=['train', 'val']))
    config = vars(parse_args())
    os.makedirs('%s' % config['name'], exist_ok=True)
    with open('%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    cudnn.benchmark = True

    # create model
    model = RFFNet(rgb_in=RGB_IN, hsi_in=N_BAND, n_classes=N_CLASS,
                   ffu_layers=2, ffu_threshold=0.1, window_size=(8, 8)).cuda()
    if config['resume']:
        checkpoint = torch.load(config['checkpoint_path'])
        model.load_state_dict(checkpoint)

    # define optimizer & scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])

    # load data
    data_ids = {"train": [], "val": []}
    for item in ["train", "val"]:
        with open(os.path.join(config["txt_root"], item + ".txt"), 'r', encoding='UTF-8') as f:
            data_ids[item] = f.readlines()
        f.close()
    print(len(data_ids["train"]), "train images:", data_ids["train"]), print(len(data_ids["val"]), "val images:", data_ids["val"])
    train_dataset = HsiRGBDataset(img_ids=[p.split('.')[0] for p in data_ids["train"]], hsi_dir=config['hsi_train'], rgb_dir=config["rgb_aligned"], mask_dir=config['mask_train'])
    val_dataset = HsiRGBDataset(img_ids=[p.split('.')[0] for p in data_ids["val"]], hsi_dir=config['hsi_val'], rgb_dir=config["rgb_aligned"], mask_dir=config['mask_val'])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True, drop_last=True)

    # load prototypes
    prototypes = []
    for root, _, files in os.walk(config['prototype_root']):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                prot = np.load(file_path)
                prototypes.append(prot[np.newaxis, ...])
    prototypes = torch.Tensor(np.vstack(prototypes)).cuda()  # [N, C, H, W]

    # train model
    best_dice, early_stop_cnt = 0, 0
    for epoch in range(config['epochs'] + 1):
        train_log = train(train_loader, epoch, model, criterion, optimizer, prototypes)
        val_log = validate(val_loader, epoch, model, criterion, prototypes)
        scheduler.step()
        print('Epoch [%d]   train-dice %.4f   val-dice %.4f' % (epoch, train_log['dice'], val_log['dice']))
        window.line([[train_log['loss'], val_log['loss']]], [epoch], win='Loss', update='append')
        window.line([[train_log["dice"], val_log["dice"]]], [epoch], win='Dice', update='append')

        # update
        if val_log['dice'] > best_dice:
            best_dice = val_log['dice']
            early_stop_cnt = 0
            tqdm.write("save model ...\n")
            torch.save(model.state_dict(), '%s/model.pkl' % config['name'])
        else:
            early_stop_cnt += 1

        if config['early_stopping_patience'] >= 0 and early_stop_cnt >= config['early_stopping_patience']:
            print(f"Early stopping triggered after epoch {epoch}.")
            break
        torch.cuda.empty_cache()


def train(train_loader, epoch, model, criterion, optimizer, prototypes):
    train_loss, train_dice = AverageMeter(), AverageMeter()
    model.train()
    tbar = tqdm(total=len(train_loader), ncols=100, position=0, leave=True)
    for i, (hsi, rgb, label, fname) in enumerate(train_loader):
        hsi, rgb, label = hsi.cuda(), rgb.cuda(), label.cuda()  # hsi: [B,1,C,H,W]   rgb: [B,C,H,W]   label: [B,H,W]
        o1, att, sim = model(rgb, hsi, prototypes)
        loss = criterion['dice_loss'](o1, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), label.shape[0])

        cal_seg_metric(o1, label, train_dice)
        descrip = 'T ({}) | loss {:.4f} |'.format(epoch, train_loss.average())
        tbar.set_description(descrip)
        tbar.update()
    tbar.close()
    return OrderedDict([('loss', train_loss.average()),
                        ('dice', train_dice.average())])


def validate(val_loader, epoch, model, criterion, prototypes):
    val_loss, val_dice = AverageMeter(), AverageMeter()
    model.eval()
    with torch.no_grad():
        tbar = tqdm(total=len(val_loader), ncols=100, position=0, leave=True)
        for i, (hsi, rgb, label, fname) in enumerate(val_loader):
            hsi, rgb, label = hsi.cuda(), rgb.cuda(), label.cuda()  # hsi: [B,1,C,H,W]   rgb: [B,C,H,W]   label: [B,H,W]
            o1, att, sim = model(rgb, hsi, prototypes)
            loss = criterion['dice_loss'](o1, label)
            val_loss.update(loss.item(), label.shape[0])
            cal_seg_metric(o1, label, val_dice)
            descrip = 'EVAL ({}) | loss {:.4f} |'.format(epoch, val_loss.average())
            tbar.set_description(descrip)
            tbar.update()

    tbar.close()
    return OrderedDict([('loss', val_loss.average()),
                        ('dice', val_dice.average())])


def test_model():
    config = vars(parse_args())

    # load data
    with open(os.path.join(config["txt_root"], "test.txt"), 'r', encoding='UTF-8') as f:
        test_ids = f.readlines()
    f.close()
    print(len(test_ids), "test images:", test_ids)
    test_dataset = HsiRGBDataset(img_ids=[p.split('.')[0] for p in test_ids],
                                 hsi_dir=config['hsi_test'], rgb_dir=config["rgb_aligned"], mask_dir=config['mask_test'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # load prototypes
    prototypes = []
    for root, _, files in os.walk(config['prototype_root']):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                prot = np.load(file_path)
                prototypes.append(prot[np.newaxis, ...])
    prototypes = torch.Tensor(np.vstack(prototypes)).cuda()  # [N, C, H, W]
    print('prototypes shape:', prototypes.shape)

    # load model
    model = RFFNet(rgb_in=RGB_IN, hsi_in=N_BAND, n_classes=N_CLASS,
                   ffu_layers=2, ffu_threshold=0.1, window_size=(8, 8)).cuda()
    model.load_state_dict(torch.load('%s/model.pkl' % config['name'], weights_only=True))

    # test model on test set
    test_loss = AverageMeter()
    labels, outs1 = [], []
    model.eval()
    with torch.no_grad():
        tbar = tqdm(total=len(test_loader), ncols=100, position=0, leave=True)
        for i, (hsi, rgb, label, fname) in enumerate(test_loader):
            hsi, rgb, label = hsi.cuda(), rgb.cuda(), label.cuda()  # hsi: [B,1,C,H,W]   rgb: [B,C,H,W]   label: [B,H,W]
            o1 = model(rgb, hsi, prototypes)
            loss = criterion['dice_loss'](o1, label)
            test_loss.update(loss.item(), label.shape[0])
            save_predict_map(o1, label, fname, folder='{}/prediction'.format(config['name']))
            outs1.extend(o1.cpu().detach().numpy())
            labels.extend(label.cpu().detach().numpy())
            tbar.set_description('TEST | loss {:.4f} |'.format(test_loss.average()))
            tbar.update()
        tbar.close()

    cal_test_metrics(np.array(outs1), np.array(labels), save_metrics=True, save_path=config['name'])
    torch.cuda.empty_cache()


if __name__ == '__main__':
    set_seed(2024)

    # main()

    test_model()
