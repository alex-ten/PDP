
def model(image_placeholders, network_layers, labels_placeholder):
    assert type(image_placeholders) is list or type(image_placeholders) is tuple
    assert type(network_layers) is list or type(network_layers) is tuple
    return {'images': image_placeholders, 'network': network_layers, 'labels': labels_placeholder}