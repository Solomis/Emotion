import cv2
import numpy as np

# OpenCV face detector
face_detector = cv2.CascadeClassifier(
    "faceAndLandmarkModels/haarcascade_frontalface_default.xml"
)

def preprocess_kdef_image(
    image_path,
    output_size=(48, 48),
    margin=0.20
):
    """
    Preprocess KDEF image to FER2013 format.

    Steps:
    1. Face detection
    2. Expand bounding box
    3. Crop face
    4. Convert to grayscale
    5. Resize to 48x48
    6. Normalize to [0,1]
    """

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not read {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        raise ValueError(
            f"No face detected in {image_path}"
        )

    # Largest detected face
    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    # Expand bounding box
    dx = int(w * margin)
    dy = int(h * margin)

    x1 = max(0, x - dx)
    y1 = max(0, y - dy)

    x2 = min(gray.shape[1], x + w + dx)
    y2 = min(gray.shape[0], y + h + dy)

    face = gray[y1:y2, x1:x2]

    # Resize to FER size
    face = cv2.resize(
        face,
        output_size,
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    face = face.astype(np.float32) / 255.0

    return face