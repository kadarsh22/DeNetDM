from module.mlp import CCIFARDeCAMModel

def get_model(model_tag, num_classes, num_layers=6):
    if model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "MLP_Product_Of_Experts":
        return MLP_Product_Of_Experts(skip_layers=3, main_layers=5, num_classes=num_classes)
    elif model_tag == 'CCIFARDeCAMModel':
        return CCIFARDeCAMModel(num_classes=num_classes)
    else:
        raise NotImplementedError