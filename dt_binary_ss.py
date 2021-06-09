import os
from pprint import pprint
import random
import sys

sys.path.insert(0, os.getcwd())
import numpy as np
import argparse
import torch
import time
from utils import check_dir, set_random_seed, accuracy, mIoU, get_logger
from models.second_segmentation import Segmentator
from data.transforms import get_transforms_binary_segmentation
from torchvision.utils import save_image
from models.pretraining_backbone import ResNet18Backbone
from data.segmentation import DataReaderBinarySegmentation
from tensorboard_evaluation import Evaluation

set_random_seed(0)
global_step = 0
DEVICE = torch.device("cuda")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        help="folder containing the data",
        default="./data_n_weights/Data/COCO_mini5class_medium",
    )
    parser.add_argument("--weights_init", type=str, default="ImageNet")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--bs", type=int, default=32, help="batch_size")
    parser.add_argument("--size", type=int, default=128, help="image size")
    parser.add_argument(
        "--snapshot-freq", type=int, default=1, help="how often to save models"
    )
    parser.add_argument(
        "--exp-suffix", type=str, default="", help="string to identify the experiment"
    )
    args = parser.parse_args()

    hparam_keys = ["lr", "bs", "size"]
    args.exp_name = "_".join(["{}{}".format(k, getattr(args, k)) for k in hparam_keys])

    args.exp_name += "_{}".format(args.exp_suffix)

    args.output_folder = check_dir(
        os.path.join(args.output_root, "dt_binseg", args.exp_name)
    )
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)
    img_size = (args.size, args.size)

    # model
    pretrained_model = ResNet18Backbone(pretrained=False).cuda()
    pretrained_model.load_state_dict(
        torch.load("./saved_models/Pretraining/saved_model.pth")["model_state"]
    )
    model = Segmentator(2, pretrained_model.features, img_size).cuda()

    # dataset
    (
        train_trans,
        val_trans,
        train_target_trans,
        val_target_trans,
    ) = get_transforms_binary_segmentation(args)
    data_root = args.data_folder
    train_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/train2014"),
        os.path.join(data_root, "aggregated_annotations_train_5classes.json"),
        transform=train_trans,
        target_transform=train_target_trans,
    )
    val_data = DataReaderBinarySegmentation(
        os.path.join(data_root, "imgs/val2014"),
        os.path.join(data_root, "aggregated_annotations_val_5classes.json"),
        transform=val_trans,
        target_transform=val_target_trans,
    )
    print("Dataset size: {} samples".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    criterion = torch.nn.BCEWithLogitsLoss().to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info("train_data {}".format(train_data.__len__()))
    logger.info("val_data {}".format(val_data.__len__()))
    epochs = 100
    epochs_no_improve = 0
    n_epochs_stop = 5
    tensorboard_eval = Evaluation(
        "./tensorboard_dir",
        "Binary_semantic",
        stats=["Loss/train", "Loss/val", "Accuracy/train", "Accuracy/val"],
    )
    best_val_loss = np.inf
    best_val_miou = 0.0
    for epoch in range(epochs):
        logger.info("Epoch {}".format(epoch))
        with torch.set_grad_enabled(True):
            epoch_train_loss, epoch_iou = train(
                train_loader, model, criterion, optimizer, logger
            )
        print(
            "Epoch: {} Train Loss : {:.4f}, Train IOU: {:.4f}".format(
                epoch, epoch_train_loss, epoch_iou
            )
        )
        with torch.no_grad():
            mean_val_loss, mean_val_iou = validate(val_loader, model, criterion, logger)
        print(
            "Epoch: {} Val Loss : {:.4f}, Val IOU: {:.4f}".format(
                epoch, mean_val_loss, mean_val_iou
            )
        )
        tb_dict = {
            "Loss/train": epoch_train_loss,
            "Loss/val": mean_val_loss,
            "Accuracy/train": epoch_iou,
            "Accuracy/val": mean_val_iou,
        }
        tensorboard_eval.write_episode_data(epoch, tb_dict)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_model(
                model,
                optimizer,
                args,
                epoch,
                mean_val_loss,
                mean_val_iou,
                logger,
                best=True,
            )
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping!")
                break


def train(loader, model, criterion, optimizer, logger):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    for _idx, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = model(inputs)
        values, indices = torch.max(outputs, dim=1, keepdim=True)
        loss = criterion(values, labels)
        logger.info("train_loss {}".format(loss.item()))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # running_iou += mIoU(indices.float(), labels.float())
    epoch_loss = running_loss / len(loader.dataset)
    # epoch_iou = running_iou / len(loader.dataset)
    epoch_iou = 0
    return epoch_loss, epoch_iou


def validate(loader, model, criterion, logger):
    model.eval()  # Set model to evaluate mode
    running_iou = 0.0
    running_loss = 0.0
    for _idx, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        values, indices = torch.max(outputs, dim=1, keepdim=True)
        loss = criterion(values, labels)
        running_loss += loss.item() * inputs.size(0)
        # running_iou += mIoU(indices.float(), labels.float())
        logger.info("val_loss {}".format(loss.item()))
    mean_val_losses = running_loss / len(loader.dataset)
    # mean_val_iou = running_iou / len(loader.dataset)
    mean_val_iou = 0
    return mean_val_losses, mean_val_iou


def save_model(model, optimizer, args, epoch, val_loss, val_iou, logger, best=False):
    # save model
    add_text_best = "BEST" if best else ""
    logger.info(
        "==> Saving "
        + add_text_best
        + " ... epoch{} loss{:.03f} miou{:.03f} ".format(epoch, val_loss, val_iou)
    )
    state = {
        "opt": args,
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": val_loss,
        "miou": val_iou,
    }
    if best:
        torch.save(state, os.path.join(args.model_folder, "ckpt_best.pth"))
    else:
        torch.save(
            state,
            os.path.join(
                args.model_folder,
                "ckpt_epoch{}_loss{:.03f}_miou{:.03f}.pth".format(
                    epoch, val_loss, val_iou
                ),
            ),
        )


if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
