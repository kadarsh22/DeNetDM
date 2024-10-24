from module.model import CCIFARDeCAMModel, CMNISTDeCAMModel, BFFHQDeCAMModel, CelebADeCAMModel


def get_model(model_tag, num_classes, stage='1'):
    if model_tag == "CMNISTDeCAMModel":
        model = CMNISTDeCAMModel(debias_hidden_layers=3, bias_hidden_layers=5, num_classes=num_classes, stage=stage)
    elif model_tag == 'CCIFARDeCAMModel':
        model = CCIFARDeCAMModel(num_classes=num_classes, stage=stage)
    elif model_tag == 'bFFHQDeCAMModel':
        assert num_classes == 2
        model = BFFHQDeCAMModel(num_classes=num_classes, stage=stage)
    elif model_tag == 'CelebADeCAMModel':
        assert num_classes == 2
        model = CelebADeCAMModel(num_classes=num_classes)
    else:
        raise NotImplementedError
    return model
