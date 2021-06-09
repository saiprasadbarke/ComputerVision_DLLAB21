import os
import torch
import random
import logging
import datetime
import numpy as np
import torch.backends.cudnn as cudnn


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mIoU(logits, gt, threshold=0.5):
    if logits.shape[1] == 1 or len(logits.shape)==3:
        if threshold==0.5:
            pred = logits.round().byte()
        else:
            pred = logits > threshold
    else:
        pred = logits.argmax(dim=1).byte()
    intersection = ((pred == 1) & (gt == 1)).sum().float()
    union = ((pred == 1) | (gt == 1)).sum().float()
    return intersection/(union+1.)


def instance_mIoU(logits, gt):
    pred = logits.argmax(dim=1).byte()
    ins_iou = []
    for instance in range(logits.shape[1]):
        if instance==0:
            continue #do not consider background
        intersection = ((pred == instance) & (gt == instance)).sum().float()
        union = ((pred == instance) | (gt == instance)).sum().float()
        if union==0:
            continue
        iou_val = intersection/(union+1.)
        ins_iou.append(iou_val)

    mean_iou = torch.mean(torch.stack(ins_iou))
    return mean_iou

def set_random_seed(seed):
    # Fix random seed to reproduce results
    # Documentation https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def get_logger(logdir, name, evaluate=False):
    # Set logger for saving process experimental information
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    logger.ts = ts
    if evaluate:
        file_path = os.path.join(logdir, "evaluate_{}.log".format(ts))
    else:
        file_path = os.path.join(logdir, "run_{}.log".format(ts))
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    # strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr = logging.StreamHandler()
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    return logger

def save_in_log(log, save_step, set="", scalar_dict=None, text_dict=None, image_dict=None):
    if scalar_dict:
        [log.add_scalar(set+"_"+k, v, save_step) for k, v in scalar_dict.items()]
    if text_dict:
        [log.add_text(set+"_"+k, v, save_step) for k, v in text_dict.items()]
    if image_dict:
        for k, v in image_dict.items():
            if k=='sample':
                log.add_images(set+"_"+k, v, save_step)
            elif k=='vec':
                log.add_images(set+"_"+k, v.unsqueeze(1).unsqueeze(1), save_step)
            elif k=='gt':
                log.add_images(set+"_"+k, v.unsqueeze(1).expand(-1, 3, -1, -1).float()/v.max(), save_step)
            elif k=='pred':
                log.add_images(set+"_"+k, v.argmax(dim=1, keepdim=True), save_step)
            elif k=='att':
                assert isinstance(v, list)
                for idx, alpha in enumerate(v):
                    log.add_images(set+"_"+k+"_"+str(idx), (alpha.unsqueeze(1)-alpha.min())/alpha.max(), save_step)
            else:
                log.add_images(set+"_"+k, v, save_step)
    log.flush()
