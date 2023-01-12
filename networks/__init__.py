import torchvision.models as models

from networks.network import *
def get_model(name, n_classes=20, version=None):
    model = _get_model_instance(name)

    if name == 'network':
        model = model()
    else:
        model = model()

    return model

def _get_model_instance(name):
    try:
        return {
            'network':network,
        }[name]
    except:
        print('Model {} not available'.format(name))
