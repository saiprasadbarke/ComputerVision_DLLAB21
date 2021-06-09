import os
import numpy as np
import argparse
import torch
from pprint import pprint
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from utils import check_dir, accuracy, get_logger
from models.pretraining_backbone import ResNet18Backbone
from tensorboard_evaluation import Evaluation

global_step = 0
DEVICE = torch.device("cuda")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        type=str,
        help="folder containing the data (crops)",
        default="./data_n_weights/Data/unlabelled_dataset/crops/images",
    )
    parser.add_argument("--weights-init", type=str, default="random")
    parser.add_argument("--output-root", type=str, default="results")
    parser.add_argument("--lr", type=float, default=1e-03, help="learning rate")
    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    parser.add_argument(
        "--size", type=int, default=128, help="size of the images to feed the network"
    )
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
        os.path.join(args.output_root, "pretrain", args.exp_name)
    )
    args.model_folder = check_dir(os.path.join(args.output_folder, "models"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # Logging to the file and stdout
    logger = get_logger(args.output_folder, args.exp_name)

    # build model and load weights
    model = ResNet18Backbone(pretrained=False).to(DEVICE)
    model.load_state_dict(
        torch.load("./data_n_weights/weights/pretrain_weights_init.pth")["model"],
        strict=False,
    )

    # load dataset
    data_root = args.data_folder
    train_transform, val_transform = get_transforms_pretraining(args)
    train_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "train"), transform=train_transform
    )
    val_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "val"), transform=val_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate,
    )

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
    )

    expdata = "  \n".join(["{} = {}".format(k, v) for k, v in vars(args).items()])
    logger.info(expdata)
    logger.info("train_data {}".format(train_data.__len__()))
    logger.info("val_data {}".format(val_data.__len__()))

    best_val_loss = np.inf
    epochs = 100
    epochs_no_improve = 0
    n_epochs_stop = 5
    tensorboard_eval = Evaluation(
        "./tensorboard_dir",
        "Pretraining",
        stats=["Loss/train", "Loss/val", "Acc/train", "Acc/val"],
    )
    # Train-validate for one epoch. You don't have to run it for 100 epochs, preferably until it starts overfitting.
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        with torch.set_grad_enabled(True):
            epoch_train_loss, epoch_train_acc = train_epoch(
                train_loader, model, criterion, optimizer
            )
        print(
            "Epoch: {} Train Loss : {:.4f}  Train Accuracy: {:.4f}".format(
                epoch, epoch_train_loss, epoch_train_acc
            )
        )
        with torch.no_grad():
            val_loss, val_acc = validate_epoch(val_loader, model, criterion)
        print(
            "Epoch: {} Val Loss : {:.4f}  Val Accuracy: {:.4f}".format(
                epoch, val_loss, val_acc
            )
        )
        tb_dict = {
            "Loss/train": epoch_train_loss,
            "Loss/val": val_loss,
            "Acc/train": epoch_train_acc,
            "Acc/val": val_acc,
        }
        tensorboard_eval.write_episode_data(epoch, tb_dict)
        # save model
        if val_loss < best_val_loss:
            current_sweet_spot = {
                "model_state": model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epochs": epoch,
                "Best train loss": epoch_train_loss,
                "Best train accuracy": epoch_train_acc,
                "Best Validation loss": val_loss,
                "Best Validation accuracy": val_acc,
            }
            torch.save(current_sweet_spot, "./saved_models/Pretraining/saved_model.pth")
            best_val_loss = val_loss
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print("Early stopping!")
                break


# train one epoch over the whole training dataset. You can change the method's signature.
def train_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    mean_training_loss = 0.0
    mean_training_accuracy = 0.0
    for idx, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        label_hat = model(inputs)
        loss = criterion(label_hat, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(label_hat, dim=1)
        running_correct += torch.sum(preds == labels.data).item()
    mean_training_loss = running_loss / len(loader.dataset)
    mean_training_accuracy = running_correct / len(loader.dataset)
    return mean_training_loss, mean_training_accuracy


# validation function. you can change the method's signature.
def validate_epoch(loader, model, criterion):
    model.eval()
    mean_val_loss = 0.0
    mean_val_accuracy = 0.0
    running_validation_loss = 0.0
    running_validation_correct = 0.0
    for _idx, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        label_hat = model(inputs)
        loss = criterion(label_hat, labels)
        running_validation_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(label_hat, dim=1)
        running_validation_correct += torch.sum(preds == labels.data).item()
        mean_val_loss = running_validation_loss / len(loader.dataset)
        mean_val_accuracy = running_validation_correct / len(loader.dataset)
    return mean_val_loss, mean_val_accuracy


if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
