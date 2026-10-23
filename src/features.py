# Shared feature extraction from teammember.ipynb, cell 0.
"""
Feature extraction shared by the Task 1 (FER2013) and Task 2 (KDEF) notebooks.

All functions take a stack of 48x48 grayscale images with pixel values in
[0, 255] (FER2013 format). KDEF images coming out of
src/preprocess.preprocess_kdef_image are in [0, 1] and must be multiplied by
255 first (see `to_fer_scale`), otherwise the landmark detector receives an
almost black image.
"""
import numpy as np
from skimage.feature import hog

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# FER2013 label order (integers 0..6). Every dataset is mapped to these ids.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# KDEF label strings (as produced by z_load_kdef_student.ipynb) -> FER ids
KDEF_TO_FER = {
    "angry": 0,
    "disgust": 1,
    "afraid": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


def fer_pixels_to_images(pixel_strings):
    """Convert the FER2013 'pixels' column (space separated) to (N, 48, 48) uint8."""
    return np.stack([
        np.array(p.split(), dtype=np.uint8).reshape(48, 48) for p in pixel_strings
    ])


def to_fer_scale(images01):
    """KDEF preprocessing returns floats in [0, 1]; FER pixels are in [0, 255]."""
    return np.clip(np.round(np.asarray(images01) * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def hog_features(images, **kwargs):
    """Histogram of Oriented Gradients. 48x48 with the defaults -> 900 dims."""
    params = {**HOG_PARAMS, **kwargs}
    return np.stack([
        hog(img.astype(np.float32) / 255.0, feature_vector=True, **params)
        for img in images
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Facial landmarks (dlib 68 points), same normalisation as the given baseline
# ---------------------------------------------------------------------------
_PREDICTORS = {}


def _get_predictor(path):
    import dlib
    if path not in _PREDICTORS:
        _PREDICTORS[path] = dlib.shape_predictor(path)
    return _PREDICTORS[path]


def landmark_points(images,
                    predictor_path="faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat"):
    """Raw (N, 68, 2) landmark coordinates in pixels. The whole 48x48 patch is
    treated as the face box, exactly as in the provided baseline."""
    import dlib
    predictor = _get_predictor(predictor_path)
    rect = dlib.rectangle(left=1, top=1, right=47, bottom=47)
    out = np.empty((len(images), 68, 2), dtype=np.float32)
    for i, img in enumerate(images):
        shape = predictor(np.ascontiguousarray(img, dtype=np.uint8), rect)
        out[i] = [(p.x, p.y) for p in shape.parts()]
    return out


def landmark_features(images,
                      predictor_path="faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat"):
    """136-dim landmark vector: centred and scaled per axis (baseline recipe)."""
    pts = landmark_points(images, predictor_path)
    # baseline: shift by box centre / box size, then centre and max-abs scale
    pts = (pts - 0.5 * 46) / 46
    pts -= pts.mean(axis=1, keepdims=True)
    pts /= np.abs(pts).max(axis=1, keepdims=True) + 1e-8
    return pts.reshape(len(images), -1).astype(np.float32)


# ---------------------------------------------------------------------------
# Raw pixels (input for PCA; the PCA itself is fitted in the notebooks so it
# only ever sees training data)
# ---------------------------------------------------------------------------
def pixel_features(images):
    """Flattened pixels scaled to [0, 1] -> 2304 dims."""
    return (np.asarray(images, dtype=np.float32) / 255.0).reshape(len(images), -1)
