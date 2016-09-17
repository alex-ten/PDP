
def model(image_placeholders, network_layers, labels_placeholder):
    model_dict = {}
    if type(image_placeholders) is list or type(image_placeholders) is tuple:
        model_dict['images'] = image_placeholders
    else:
        model_dict['images'] = [image_placeholders]

    if type(network_layers) is list or type(network_layers) is tuple:
        model_dict['network'] = network_layers
    else:
        model_dict['network'] = [network_layers]
    model_dict['labels'] = labels_placeholder
    return model_dict