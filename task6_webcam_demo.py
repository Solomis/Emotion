"""Task 6: live facial-expression recognition with the saved Task 5 model.

Run --help for baseline, static-image, video, and optional smoothing modes.
Only load trusted course/locally trained pickle files: joblib loads Python objects.
"""
import argparse
from collections import Counter, deque
import time
import numpy as np
import joblib
from src.emotion_features import (ROOT, LABELS, LANDMARK_CONFIG,
                                  extract_model_features as _independent_features, load_predictor)


def extract_model_features(image, config, predictor=None):
    """Extract the features specified by the saved model."""
    # The course reference and archived models use the original extractor.
    if config.get('implementation') is None:
        return _independent_features(image, config, predictor)
    from src.teammate_features import extract_model_features as extract_features
    return extract_features(image, config, predictor)


def load_model_and_preprocessing(path):
    """Load the fitted model and its saved preprocessing settings."""
    path = ROOT / path
    artifact = joblib.load(path)

    # The lecturer's reference file stores a bare landmark classifier.
    if not isinstance(artifact, dict):
        artifact = dict(estimator=artifact,
                        feature_config=dict(kind='landmarks', landmarks=LANDMARK_CONFIG),
                        labels=LABELS, input_shape=(48, 48), input_dtype='uint8')

    # One face at a time does not need a Random Forest worker pool.
    estimator = artifact['estimator']
    parameters = estimator.get_params()
    if 'classifier__n_jobs' in parameters:
        estimator.set_params(classifier__n_jobs=1)
    elif 'n_jobs' in parameters:
        estimator.set_params(n_jobs=1)
    return artifact


def preprocess_face(gray, box=None):
    """Clip/crop a grayscale uint8 image and resize to (48,48), preserving pixel range."""
    # Clip the detector rectangle before resizing the face to the training resolution.
    gray = np.asarray(gray)
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size == 0:
        raise ValueError('Expected a nonempty grayscale uint8 frame.')
    if box is not None:
        x,y,w,h = map(int,box)
        if w <= 0 or h <= 0:
            raise ValueError('Nonpositive face dimensions.')
        x1,y1=max(0,x),max(0,y)
        x2,y2=min(gray.shape[1],x+w),min(gray.shape[0],y+h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Face outside frame.')
        gray=gray[y1:y2,x1:x2]
    if gray.shape == (48, 48):
        return gray.copy()
    import cv2
    return cv2.resize(gray,(48,48),interpolation=cv2.INTER_LINEAR)


def predict_emotion(image, artifact, predictor=None, features=None):
    """Return the predicted expression ID for one prepared face."""
    if features is None:
        features = extract_model_features(image, artifact['feature_config'], predictor)
    # scikit-learn expects a batch: one row containing this face's features.
    row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    # The pipeline applies its fitted scalers before the classifier; nothing is refitted.
    return int(artifact['estimator'].predict(row)[0])


def detect_faces(gray, cascade):
    """Return Haar face rectangles (N,4) for a grayscale camera frame."""
    return cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)


def initialize_camera(source):
    """Open a camera index or video path and return its OpenCV capture object."""
    import cv2
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot open camera/video {source}; check permissions and camera index.')
    return cap


def run_webcam_demo(artifact, camera=0, video=None, smooth=1, max_frames=None):
    """Run face detection/prediction/display until q, video EOF, or a frame limit."""
    import cv2
    cascade=cv2.CascadeClassifier(str(ROOT / 'faceAndLandmarkModels/haarcascade_frontalface_default.xml'))
    if cascade.empty():
        raise RuntimeError('Missing or invalid Haar cascade XML.')
    config=artifact['feature_config']
    predictor=load_predictor(config['landmarks']) if config['kind'] in ('landmarks','combined','geometry','augmented') else None
    cap=initialize_camera(str(ROOT / video) if video else camera)
    history = deque(maxlen=smooth)
    previous = None
    frames = detections = failures = 0
    raw_counts = Counter()
    displayed_counts = Counter()
    face_frames = max_faces = 0
    raw_switches = displayed_switches = transitions = 0
    last_raw = last_displayed = frame_shape = None
    started=time.perf_counter()
    try:
        while True:
            tick = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                if video:
                    break
                raise RuntimeError('Camera stopped returning valid frames.')
            # Locate faces first; the classifier only sees cropped grayscale faces.
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame_shape=gray.shape
            faces=detect_faces(gray,cascade)
            face_frames += int(len(faces) > 0)
            max_faces = max(max_faces, len(faces))
            # Do not carry a person's voting history into another face or a detection gap.
            if len(faces) != 1:
                history.clear()
                previous = last_raw = last_displayed = None
            for box in faces:
                x,y,w,h=map(int,box)
                try:
                    image=preprocess_face(gray,box)
                    # Reuse the loaded landmark predictor for all faces and frames.
                    features=extract_model_features(image,config,predictor)
                except (ValueError,RuntimeError) as exc:
                    failures += 1
                    history.clear()
                    previous = last_raw = last_displayed = None
                    print(f'Feature/crop failure {failures}: {exc}',flush=True)
                    cv2.putText(frame,'Feature extraction failed',(max(0,x),max(20,y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
                    continue
                label = predict_emotion(image, artifact, predictor, features)
                detections += 1
                raw = label
                raw_counts[LABELS[raw]] += 1
                # Vote only while a single face stays in roughly the same position.
                if len(faces) == 1:
                    if previous is not None:
                        px,py,pw,ph=previous
                        intersection=max(0,min(x+w,px+pw)-max(x,px))*max(0,min(y+h,py+ph)-max(y,py))
                        union=w*h+pw*ph-intersection
                        if intersection/max(1,union) < .5:
                            history.clear()
                            last_raw = last_displayed = None
                    history.append(label)
                    previous = (x, y, w, h)
                    counts=Counter(history)
                    # A tie prefers the newest tied prediction, limiting lag.
                    label=max(reversed(history),key=lambda p:counts[p])
                    if last_raw is not None:
                        transitions+=1
                        raw_switches+=int(raw != last_raw)
                        displayed_switches+=int(label != last_displayed)
                    last_raw = raw
                    last_displayed = label
                displayed_counts[LABELS[label]]+=1
                # Draw the displayed prediction over the original colour frame.
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(frame,LABELS[label],(max(0,x),max(20,y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,0),2)
            frames+=1
            fps=1/max(1e-9,time.perf_counter()-tick)
            status=f'{fps:.1f} processing FPS | q: quit | vote window {smooth}'
            if not len(faces):
                status += ' | No face detected'
            cv2.putText(frame,status,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow('Task 6 - Facial expression recognition',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        # Always release the webcam, including when the window closes after an error.
        cap.release()
        cv2.destroyAllWindows()
    seconds=time.perf_counter()-started
    return dict(frames=frames,face_predictions=detections,feature_failures=failures,
                elapsed_seconds=seconds,throughput_fps=frames/max(seconds,1e-9),
                face_detected_frames=face_frames,max_faces=max_faces,frame_shape=frame_shape,
                raw_label_counts=dict(raw_counts),displayed_label_counts=dict(displayed_counts),
                continuous_single_face_transitions=transitions,
                raw_label_switches=raw_switches,displayed_label_switches=displayed_switches)


def main():
    """Parse command-line options and run a static check or live demo; return exit status."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='models/task5_best_model.pkl')
    parser.add_argument('--lecturer',action='store_true')
    parser.add_argument('--camera',type=int,default=0)
    parser.add_argument('--video',help='Optional prerecorded video instead of camera')
    parser.add_argument('--image',help='Already cropped face image; predict without a GUI')
    parser.add_argument('--smooth',type=int,default=1,help='Majority window; 1 disables smoothing')
    parser.add_argument('--max-frames',type=int)
    args=parser.parse_args()
    if args.smooth < 1:
        parser.error('--smooth must be positive')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    try:
        artifact=load_model_and_preprocessing('models/rf_fer_model_landmarks.pkl' if args.lecturer else args.model)
        if args.image:
            import cv2
            image=cv2.imread(str(ROOT / args.image),cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f'Cannot read image: {args.image}')
            label=predict_emotion(preprocess_face(image),artifact)
            print(f'{label}: {LABELS[label]}')
        else:
            print(run_webcam_demo(artifact,args.camera,args.video,args.smooth,args.max_frames))
        return 0
    except (OSError,ValueError,RuntimeError,ImportError) as exc:
        parser.exit(1,f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
