from module.model import CCIFARDeNetDMModel, CMNISTDeNetDMModel, BFFHQDeNetDModel, CelebADeNetDMModel


def get_model(model_tag, num_classes, stage='1'):
    if model_tag == "CMNISTDeNetDMModel":
        model = CMNISTDeNetDMModel(debias_hidden_layers=3, bias_hidden_layers=5, num_classes=num_classes, stage=stage)
    elif model_tag == 'CCIFARDeNetDMModel':
        model = CCIFARDeNetDMModel(num_classes=num_classes, stage=stage)
    elif model_tag == 'bFFHQDeNetDModel':
        assert num_classes == 2
        model = BFFHQDeNetDModel(num_classes=num_classes, stage=stage)
    elif model_tag == 'CelebADeNetDMModel':
        assert num_classes == 2
        model = CelebADeNetDMModel(num_classes=num_classes)
    else:
        raise NotImplementedError
    return model
