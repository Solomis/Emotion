"""Single-image adapter for the shared batch feature functions.

The notebooks and camera use this adapter for the group pipeline. Legacy models use the extractor in emotion_features.py.
"""
from pathlib import Path
import numpy as np
from src import features as teammate
from src.emotion_features import (validate_image, geometry_features,
                                  LABELS, LANDMARK_CONFIG, ROOT)

HOG_CONFIG = {**teammate.HOG_PARAMS, 'transform_sqrt': False}
IMPLEMENTATION = 'teammate-v1'
FEATURE_SOURCES = ['src/features.py', 'src/teammate_features.py', 'src/emotion_features.py']


def extract_model_features(image, config, predictor=None):
    """Return finite float32 (D,) features for one (48,48) uint8 face.

    HOG uses float32 pixels divided by 255 without square-root compression.
    Landmarks retain the baseline normalization epsilon and coordinate ordering. Pixels remain /255; learned PCA
    belongs to the saved estimator. Augmented vectors append selected geometric
    ratios in the saved order. predictor optionally reuses a loaded dlib model.
    Bad shapes/nonfinite output raise rather than producing a prediction.
    """
    image=validate_image(image)
    kind=config['kind']
    if kind=='hog':
        values=teammate.hog_features(image[None,:,:],**config['hog'])[0]
    elif kind=='landmarks':
        path=str((ROOT/config['landmarks']['predictor']).resolve())
        if predictor is not None: teammate._PREDICTORS[path]=predictor
        values=teammate.landmark_features(image[None,:,:],predictor_path=path)[0]
    elif kind=='pca':
        values=teammate.pixel_features(image[None,:,:])[0]
    elif kind=='combined':
        values=np.concatenate([
            extract_model_features(image,dict(kind='hog',hog=config['hog']),predictor),
            extract_model_features(image,dict(kind='landmarks',landmarks=config['landmarks']),predictor)])
    elif kind=='geometry':
        values=geometry_features(image,config['landmarks'],predictor)
    elif kind=='augmented':
        values=np.concatenate([extract_model_features(image,config['base'],predictor),
            geometry_features(image,config['landmarks'],predictor)[config['geometry_indices']]])
    else:
        raise ValueError(f'Unsupported teammate representation: {kind}')
    values=np.asarray(values,dtype=np.float32)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError('Teammate feature extractor returned invalid features.')
    return values
