

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from tqdm import tqdm

os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import (
    HAND_DIM, FACE_DIM, BODY_DIM, TOTAL_DIM,
    FACE_KEYPOINT_INDICES, BODY_KEYPOINT_INDICES,
    SEQUENCE_LENGTH, NUM_NODES,
)

mp_holistic = mp.solutions.holistic



def _hand_to_array(landmarks) -> np.ndarray:
    """21 hand landmarks → R^63.  Returns zeros if None."""
    if landmarks is None:
        return np.zeros(63, dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark],
                    dtype=np.float32).flatten()


def _face_to_array(landmarks) -> np.ndarray:
    """40 salient face landmarks → R^120.  Returns zeros if None."""
    if landmarks is None:
        return np.zeros(FACE_DIM, dtype=np.float32)
    lms = landmarks.landmark
    pts = [[lms[i].x, lms[i].y, lms[i].z] for i in FACE_KEYPOINT_INDICES]
    return np.array(pts, dtype=np.float32).flatten()


def _body_to_array(landmarks) -> np.ndarray:
    """11 upper-body pose landmarks → R^33.  Returns zeros if None."""
    if landmarks is None:
        return np.zeros(BODY_DIM, dtype=np.float32)
    lms = landmarks.landmark
    pts = [[lms[i].x, lms[i].y, lms[i].z] for i in BODY_KEYPOINT_INDICES]
    return np.array(pts, dtype=np.float32).flatten()


def _results_to_vector(res) -> np.ndarray:
    """Convert already-processed MediaPipe results to 279-D vector."""
    left_hand  = _hand_to_array(res.left_hand_landmarks)    # R^63
    right_hand = _hand_to_array(res.right_hand_landmarks)   # R^63
    face       = _face_to_array(res.face_landmarks)          # R^120
    body       = _body_to_array(res.pose_landmarks)          # R^33
    return np.concatenate([left_hand, right_hand, face, body])  # R^279



def extract_frame_landmarks(frame_bgr: np.ndarray,
                             holistic,
                             results=None) -> np.ndarray:
    
    if results is not None:
        return _results_to_vector(results)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = holistic.process(rgb)
    return _results_to_vector(res)


def extract_frame_landmarks_structured(
        frame_bgr: np.ndarray,
        holistic,
        results=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    if results is None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)
    else:
        res = results

    def _pts(lms, n):
        if lms is None:
            return np.zeros((n, 3), dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in lms.landmark],
                        dtype=np.float32)

    left  = _pts(res.left_hand_landmarks,  21)   # (21,3)
    right = _pts(res.right_hand_landmarks, 21)   # (21,3)
    hand_pts = np.vstack([left, right])           # (42,3)

    if res.face_landmarks is not None:
        lms = res.face_landmarks.landmark
        face_pts = np.array([[lms[i].x, lms[i].y, lms[i].z]
                              for i in FACE_KEYPOINT_INDICES],
                             dtype=np.float32)    # (40,3)
    else:
        face_pts = np.zeros((40, 3), dtype=np.float32)

    if res.pose_landmarks is not None:
        lms = res.pose_landmarks.landmark
        body_pts = np.array([[lms[i].x, lms[i].y, lms[i].z]
                              for i in BODY_KEYPOINT_INDICES],
                             dtype=np.float32)    # (11,3)
    else:
        body_pts = np.zeros((11, 3), dtype=np.float32)

    return hand_pts, face_pts, body_pts



def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    seq : (T, 279)
    Centers each frame at its mean across active (non-zero) landmarks,
    then normalizes to unit variance across the sequence.
    """
    seq = seq.copy()
    nonzero_mask = np.any(seq != 0, axis=1)
    if nonzero_mask.sum() == 0:
        return seq
    mean = seq[nonzero_mask].mean(axis=0, keepdims=True)
    std  = seq[nonzero_mask].std(axis=0, keepdims=True) + 1e-6
    seq[nonzero_mask] = (seq[nonzero_mask] - mean) / std
    return seq



def extract_sequence_from_folder(
        folder: str,
        seq_len: int = SEQUENCE_LENGTH,
        holistic_kwargs: Optional[dict] = None) -> Optional[np.ndarray]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower() in valid_ext])
    if not frames:
        return None

    hkw = holistic_kwargs or {}
    kps = []
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        **hkw,
    ) as h:
        for fname in frames:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            kps.append(extract_frame_landmarks(img, h))

    if not kps:
        return None

    arr = np.array(kps, dtype=np.float32)   # (N, 279)
    arr = _resample(arr, seq_len)
    arr = normalize_sequence(arr)
    return arr


def extract_sequence_from_video(
        video_path: str,
        seq_len: int = SEQUENCE_LENGTH) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    kps = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as h:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            kps.append(extract_frame_landmarks(frame, h))
    cap.release()

    if not kps:
        return None

    arr = np.array(kps, dtype=np.float32)
    arr = _resample(arr, seq_len)
    arr = normalize_sequence(arr)
    return arr


def _resample(arr: np.ndarray, target: int) -> np.ndarray:
    """Uniformly subsample or pad with zeros to reach target length."""
    n = len(arr)
    if n == target:
        return arr
    elif n > target:
        idx = np.linspace(0, n - 1, target, dtype=int)
        return arr[idx]
    else:
        pad = np.zeros((target - n, arr.shape[1]), dtype=np.float32)
        return np.vstack([arr, pad])

def _sliding_window_sequences(frames_kps: np.ndarray,
                               seq_len: int,
                               stride: int) -> list:
    N = len(frames_kps)
    seqs = []

    if N <= seq_len:
        arr = _resample(frames_kps, seq_len)
        seqs.append(normalize_sequence(arr))
        return seqs

    for start in range(0, N - seq_len + 1, stride):
        window = frames_kps[start: start + seq_len]
        seqs.append(normalize_sequence(window.copy()))

    if (N - seq_len) % stride != 0:
        window = frames_kps[N - seq_len:]
        seqs.append(normalize_sequence(window.copy()))

    return seqs


def bulk_extract(root_dir: str,
                 seq_len: int = SEQUENCE_LENGTH,
                 cache_path: Optional[str] = None,
                 stride: Optional[int] = None):
    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Loading from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["sequences"], data["labels"].tolist(), data["label_map"].item()

    if stride is None:
        stride = max(1, seq_len // 4)

    valid_img = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])
    label_map  = {name: idx for idx, name in enumerate(class_dirs)}
    sequences, labels = [], []

    print(f"[Extract] {len(class_dirs)} classes in {root_dir}  (stride={stride})")

    for cls in tqdm(class_dirs, desc="Classes"):
        cls_path = os.path.join(root_dir, cls)
        sub_dirs = sorted([s for s in os.listdir(cls_path)
                           if os.path.isdir(os.path.join(cls_path, s))])

        if sub_dirs:
            for sdir in sub_dirs:
                sample_path = os.path.join(cls_path, sdir)
                frame_files = sorted([
                    f for f in os.listdir(sample_path)
                    if os.path.splitext(f)[1].lower() in valid_img
                ])
                if not frame_files:
                    continue

                kps = _extract_kps_from_files(sample_path, frame_files)
                if kps is None:
                    continue

                for seq in _sliding_window_sequences(kps, seq_len, stride):
                    sequences.append(seq)
                    labels.append(label_map[cls])
        else:
            frame_files = sorted([
                f for f in os.listdir(cls_path)
                if os.path.splitext(f)[1].lower() in valid_img
            ])
            if not frame_files:
                continue

            kps = _extract_kps_from_files(cls_path, frame_files)
            if kps is None:
                continue

            for seq in _sliding_window_sequences(kps, seq_len, stride):
                sequences.append(seq)
                labels.append(label_map[cls])

    sequences = np.array(sequences, dtype=np.float32)
    print(f"[Extract] Done — {len(sequences)} sequences, shape {sequences.shape}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path,
                            sequences=sequences,
                            labels=np.array(labels),
                            label_map=label_map)
        print(f"[Cache] Saved to {cache_path}")

    return sequences, labels, label_map


def bulk_extract_images(root_dir: str,
                        seq_len: int = SEQUENCE_LENGTH,
                        cache_path: Optional[str] = None,
                        augment_copies: int = 3):
    if cache_path and os.path.exists(cache_path):
        print(f"[Cache] Loading from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["sequences"], data["labels"].tolist(), data["label_map"].item()

    valid_img = {".jpg", ".jpeg", ".png", ".bmp"}
    class_dirs = sorted([d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d))])
    label_map  = {name: idx for idx, name in enumerate(class_dirs)}
    sequences, labels = [], []

    print(f"[Extract-Images] {len(class_dirs)} classes in {root_dir}")

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3,
    ) as h:
        for cls in tqdm(class_dirs, desc="Classes"):
            cls_path   = os.path.join(root_dir, cls)
            img_files  = sorted([
                f for f in os.listdir(cls_path)
                if os.path.splitext(f)[1].lower() in valid_img
            ])
            if not img_files:
                continue

            for fname in img_files:
                img = cv2.imread(os.path.join(cls_path, fname))
                if img is None:
                    continue

                kp = extract_frame_landmarks(img, h)
                _add_tiled_sequence(kp, seq_len, sequences, labels,
                                    label_map[cls])

                for _ in range(augment_copies - 1):
                    aug = _augment_image(img)
                    kp_aug = extract_frame_landmarks(aug, h)
                    _add_tiled_sequence(kp_aug, seq_len, sequences, labels,
                                        label_map[cls])

    sequences = np.array(sequences, dtype=np.float16)
    print(f"[Extract-Images] Done — {len(sequences)} sequences, "
          f"shape {sequences.shape}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path,
                            sequences=sequences,
                            labels=np.array(labels),
                            label_map=label_map)
        print(f"[Cache] Saved to {cache_path}")

    return sequences, labels, label_map


def _add_tiled_sequence(kp: np.ndarray, seq_len: int,
                         sequences: list, labels: list, label_idx: int):
    if np.all(kp == 0):
        return
    seq = np.tile(kp[np.newaxis, :], (seq_len, 1))
    seq = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.005
    seq = normalize_sequence(seq)
    sequences.append(seq)
    labels.append(label_idx)


def _augment_image(img: np.ndarray) -> np.ndarray:
    import random
    h, w = img.shape[:2]

    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    alpha = random.uniform(0.8, 1.2)
    beta  = random.randint(-20, 20)
    img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    angle = random.uniform(-10, 10)
    M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img   = cv2.warpAffine(img, M, (w, h),
                            borderMode=cv2.BORDER_REFLECT_101)
    return img


def _extract_kps_from_files(folder: str,
                             frame_files: list) -> Optional[np.ndarray]:
    kps = []
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as h:
        for fname in frame_files:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            kps.append(extract_frame_landmarks(img, h))
    if not kps:
        return None
    return np.array(kps, dtype=np.float32)