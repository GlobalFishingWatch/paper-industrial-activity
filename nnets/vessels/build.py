"""Build and compile model.

Compile from module (w/weights, optional), or
Load full model from binary store

"""
import importlib

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

module_dir = "classification/vessels"


def get_model(name=None, path=None, params=None):
    """Build a model with associated weights.

    Parameters
    ----------
    name : string
        Name of the module defining the model.
    path : string
        Path to stored binary model.
    params : namedtuple
        Parameters for the model (e.g. params.groups).

    Returns
    -------
    subclass of KerasModel

    """
    if name:
        modname = f"{module_dir}/{name}".replace("/", ".")
        modelmod = importlib.import_module(modname)

        model = modelmod.build_model(
            base_filter_count=params.base_filters,
            tile_size=params.tile_size,
            groups=params.groups,
            keep_prob=1.0,
            l2=1e-6,
        )

        opt = SGD(momentum=0.9, clipnorm=1.0)
        losses = {
            "presence": BinaryCrossentropy(),
            "length": modelmod.squared_error,
        }
        metrics = {
            "presence": BinaryAccuracy(),
            "length": modelmod.squared_error,
        }
        loss_weights = {"presence": 1, "length": 0.01}

        model.compile(
            optimizer=opt,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights,
        )

    elif path:
        model = load_model(path)

    else:
        raise ("Need name + params or path")

    return model
