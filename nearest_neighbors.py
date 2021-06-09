import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from torchvision.utils import save_image
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg


DEVICE = torch.device("cuda")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-init", type=str, default="")
    parser.add_argument(
        "--size", type=int, default=256, help="size of the images to feed the network"
    )
    parser.add_argument("--output-root", type=str, default="results")
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(
            args.output_root,
            "nearest_neighbors",
            args.weights_init.replace("/", "_").replace("models", ""),
        )
    )
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    model = ResNet18Backbone(pretrained=False).to(DEVICE)
    # TODO: Load model correctly check if weights are needed to be loaded separately
    saved_sweetspot = torch.load("./saved_models/Pretraining/saved_model.pth")
    model.load_state_dict(saved_sweetspot["model_state"])
    print(f"Model pretrained till Epoch: {saved_sweetspot['epochs']}")
    # dataset
    data_root = "./data_n_weights/Data/unlabelled_dataset/crops/images/"
    val_transform = Compose(
        [Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()]
    )
    val_data = DataReaderPlainImg(
        os.path.join(data_root, str(args.size), "val"), transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    # query_indices = random.sample(range(0, 1000), 10)
    query_indices = [15, 861, 49, 367, 473, 11, 966, 914, 462, 371]
    print(f"Queried indices are {query_indices}")
    for index in query_indices:
        os.makedirs(f"{args.output_folder}/{index}")
    nns = []
    for idx, img in enumerate(val_loader):

        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, _ = find_nn(model, img, val_loader, 10)
        for query_idx, query_img in enumerate(val_loader):
            if query_idx in closest_idx:
                save_image(
                    query_img,
                    f"{args.output_folder}{idx}/{closest_idx.index(query_idx)}_{query_idx}.jpg",
                )


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
        model: the model for computing the features
        query_img: the image of which to find the NNs
        loader: the loader for the dataset in which to look for the NNs
        k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    closest_dist = []
    with torch.no_grad():
        model.eval()
        learnt_features = model(query_img.to(DEVICE))
        distances = {}
        for idx, img in enumerate(loader):
            learnt_features_to_be_compared = model(img.cuda())
            distances[idx] = torch.cdist(
                learnt_features, learnt_features_to_be_compared
            ).item()
        distances = sorted(distances.items(), key=lambda item: item[1])
        closest_idx, closest_dist = zip(*distances)
    return closest_idx[:k], closest_dist[:k]


if __name__ == "__main__":
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args)
