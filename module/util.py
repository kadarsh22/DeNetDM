from module.mlp import CCIFARDeCAMModel, CMNISTDeCAMModel

def get_model(model_tag, num_classes):
    if model_tag == "CMNISTDeCAMModel":
        return CMNISTDeCAMModel(debias_layers=3, bias_layers=5, num_classes=num_classes)
    elif model_tag == 'CCIFARDeCAMModel':
        return CCIFARDeCAMModel(num_classes=num_classes)
    else:
        raise NotImplementedError