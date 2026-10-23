"""Deterministic features shared by the assignment notebooks and webcam.

Only feature extraction lives here; fitting and experiment selection stay in notebooks.
Inputs are already cropped FER-style uint8 images, never normalized KDEF floats.
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
HOG_CONFIG = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)
LANDMARK_CONFIG = dict(rectangle=(1, 1, 47, 47), normalization='center_axis_maxabs',
                       predictor='faceAndLandmarkModels/shape_predictor_68_face_landmarks_GTX.dat')


def validate_image(image):
    """Return an unchanged (48,48) uint8 image or raise ValueError.

    Strict dtype/shape checks prevent silently treating [0,1] floats as [0,255]
    pixels. Cropping/resizing belongs to the camera adapter, not this function.
    """
    image = np.asarray(image)
    if image.shape != (48, 48) or image.dtype != np.uint8:
        raise ValueError('Expected a (48,48) uint8 grayscale image in [0,255].')
    return image


def load_predictor(config):
    """Load dlib's 68-point predictor from feature metadata; missing files raise.

    config is a landmark configuration dict. The returned predictor is reusable
    across images so the expensive model is loaded only once per extraction run.
    """
    import dlib
    path = ROOT / config['predictor']
    if not path.is_file():
        raise FileNotFoundError(path)
    return dlib.shape_predictor(str(path))


def landmark_points(image, config=None, predictor=None):
    """Return raw (68,2) float32 points in dlib order for a FER-style image.

    The fixed rectangle reproduces the supplied training notebook, rather than
    running a second face detector on an already cropped face. Runtime failures
    propagate; finite coordinates alone cannot establish anatomical correctness.
    """
    import dlib
    config = LANDMARK_CONFIG if config is None else config
    predictor = load_predictor(config) if predictor is None else predictor
    shape = predictor(validate_image(image), dlib.rectangle(*config['rectangle']))
    points = np.asarray([(p.x, p.y) for p in shape.parts()], dtype=np.float32)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid landmark output.')
    return points


def extract_model_features(image, config, predictor=None):
    """Convert one (48,48) uint8 image to a finite 1-D float32 feature vector.

    config contains kind ('hog', 'landmarks', 'pca', 'combined', 'geometry',
    'augmented') and parameters. Augmented appends selected geometric ratios to
    the base vector; selection indices come from KDEF Training identities only.
    PCA input is 2304 pixels /255; the learned PCA lives in the saved Pipeline.
    Combined order is HOG then landmarks. Degenerate landmarks raise ValueError
    instead of dividing by zero or fabricating valid geometry.
    """
    image = validate_image(image)
    kind = config['kind']
    if kind == 'pca':
        features = image.ravel().astype(np.float32) / 255.0
    elif kind == 'hog':
        from skimage.feature import hog
        features = hog(image, **config['hog'])
    elif kind == 'landmarks':
        settings = config['landmarks']
        if settings['normalization'] != 'center_axis_maxabs':
            raise ValueError('Unsupported landmark normalization.')
        points = landmark_points(image, settings, predictor)
        left, top, right, bottom = settings['rectangle']
        # Preserve the original floating-point operations and x,y interleaving.
        widths = np.array([right-left, bottom-top], dtype=np.float32)
        points = (points - 0.5 * widths) / widths
        points -= points.mean(axis=0, keepdims=True)
        scale = np.max(np.abs(points), axis=0, keepdims=True)
        if np.any(scale <= np.finfo(np.float32).eps):
            raise ValueError('Degenerate landmark geometry.')
        features = (points / scale).ravel()
    elif kind == 'combined':
        features = np.concatenate([
            extract_model_features(image, dict(kind='hog', hog=config['hog']), predictor),
            extract_model_features(image, dict(kind='landmarks', landmarks=config['landmarks']), predictor)])
    elif kind == 'geometry':
        features = geometry_features(image, config['landmarks'], predictor)
    elif kind == 'augmented':
        features = np.concatenate([
            extract_model_features(image, config['base'], predictor),
            geometry_features(image, config['landmarks'], predictor)[config['geometry_indices']]])
    else:
        raise ValueError(f'Unknown feature kind: {kind}')
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError('Non-finite features.')
    return features


GEOMETRY_NAMES = ['mouth_opening', 'mouth_width', 'left_eye_opening',
                  'right_eye_opening', 'left_brow_eye_gap', 'right_brow_eye_gap']


def geometry_features(image, config=None, predictor=None):
    """Return six float32 distance ratios from one FER-style face, shape (6,).

    Mouth/lid/brow distances are divided by the distance between eye centers to
    reduce dependence on face size. Unlike axis-wise landmark normalization,
    this keeps relative vertical/horizontal geometry. Fixed dlib point indices
    are zero based. Degenerate eye separation raises ValueError; no ratio is
    evidence of a person's internal emotional state.
    """
    points = landmark_points(image, config, predictor)
    eye_distance = np.linalg.norm(points[36:42].mean(0) - points[42:48].mean(0))
    if eye_distance <= np.finfo(np.float32).eps:
        raise ValueError('Degenerate eye-center distance.')
    pairs = [(62,66),(48,54),(37,41),(43,47),(19,37),(24,44)]
    return np.array([np.linalg.norm(points[a]-points[b])/eye_distance
                     for a,b in pairs],dtype=np.float32)
