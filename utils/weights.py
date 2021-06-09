import torch

# Load from weights
def load_from_weights(model, weights, logger=None):
    if logger:
        logger.info("Loading weights from "+weights)
    else:
        print("Loading weights from "+weights)

    # Get weights from saved file and check in model
    ckpt = torch.load(weights, map_location='cpu')
    loaded_dict = {k.replace("module.", "").replace("features.", ""): v for k, v in ckpt['model'].items()}
    model_dict = model.state_dict()
    pretrained_dict = {}
    weights_ignore = []
    for k, v in loaded_dict.items():
        if k in model_dict.keys():
            match_size = (v.shape==model_dict[k].shape)
            if match_size:
                pretrained_dict[k] = v
            else:
                weights_ignore.append(k)
        else:
            weights_ignore.append(k)
    expdata = "  ".join(["{}".format(k) for k in weights_ignore])

    if logger:
        logger.info('Weights ignored from loaded model: '+expdata)
    else:
        print('Weights ignored from loaded model: '+expdata)

    weights_ignore = []
    for k, v in model_dict.items():
        if k not in pretrained_dict.keys():
            weights_ignore.append(k)
    expdata = "  ".join(["{}".format(k) for k in weights_ignore])
    if logger:
        logger.info('Weights ignored from training model: '+expdata)
    else:
        print('Weights ignored from training model: '+expdata)

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if logger:
        logger.info("Done loading pretrained weights")
    else:
        print("Done loading pretrained weights")

    return model
