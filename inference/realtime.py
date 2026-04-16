import os, sys, time, collections, argparse, json
import cv2
import numpy as np
import torch
import mediapipe as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import (
    DEVICE, CHECKPOINT_DIR, CACHE_DIR,
    SEQUENCE_LENGTH, TOTAL_DIM, PREDICTION_THRESH, WEBCAM_INDEX,
)
from src.landmark_extractor import (
    extract_frame_landmarks,
    normalize_sequence,
)
from src.models.landmark_fusion_st import LandmarkFusionST
from src.models.semantic_refiner import SemanticRefiner, GlossVocab

mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles


#  Model loading helpers
def _detect_num_classes(ckpt_path: str) -> int:
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    for key in reversed(list(sd.keys())):
        if "weight" in key and sd[key].dim() == 2:
            return sd[key].shape[0]
    return 0


def load_model(ckpt_path: str, num_classes: int = 0,
               use_ctc: bool = False) -> LandmarkFusionST:
    if num_classes == 0:
        num_classes = _detect_num_classes(ckpt_path)
    model = LandmarkFusionST(num_classes, dropout=0.0,
                              use_ctc=use_ctc).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()
    print(f"[Model] Loaded {ckpt_path}  ({num_classes} classes)")
    return model


def load_label_map(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    data      = np.load(cache_path, allow_pickle=True)
    label_map = data["label_map"].item()          # {name: idx}
    inv_lm    = {v: k for k, v in label_map.items()}
    return inv_lm                                 # {idx: name}  — direct, no remap


# Prediction 
@torch.no_grad()
def predict(model: LandmarkFusionST,
            buffer: np.ndarray,
            idx2label: dict) -> tuple:
    seq    = normalize_sequence(buffer.copy())
    x      = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    logits, _ = model(x)                          # (1, num_classes)
    probs  = torch.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    label  = idx2label.get(idx.item(), f"cls_{idx.item()}")
    return label, conf.item()


#  Draw landmarks from already-processed results 

def draw_landmarks(display, results):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            display, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style())
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            display, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style())
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            display, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            display, results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw_styles
                .get_default_face_mesh_contours_style())


#  Main inference loop 

def run(word_ckpt: str = None, sentence_ckpt: str = None):
    print("=" * 65)
    print("  LandmarkFusion-ST  —  Real-Time Inference")
    print("=" * 65)
    print("  Controls:  q=quit | m=toggle mode | c=clear history | s=save landmarks")
    print("  Full pipeline: MediaPipe → GAT → TCN+MHSA → CTC/classification.")
    print(f"  Confidence threshold: {PREDICTION_THRESH:.2f}  (high-conf shown GREEN, low-conf shown YELLOW, all shown GREY)")

    #  Load models 
    word_model = sentence_model = None
    word_labels = sentence_labels = {}

    word_cache     = os.path.join(CACHE_DIR, "word_seq_cache.npz")
    sentence_cache = os.path.join(CACHE_DIR, "isl_seq_cache.npz")

    if word_ckpt and os.path.exists(word_ckpt):
        word_model  = load_model(word_ckpt, use_ctc=False)
        word_labels = load_label_map(word_cache)

    if sentence_ckpt and os.path.exists(sentence_ckpt):
        sentence_model  = load_model(sentence_ckpt, use_ctc=False)
        sentence_labels = load_label_map(sentence_cache)

    if word_model is None:
        ckpt = os.path.join(CHECKPOINT_DIR, "best_lmfst_word.pth")
        if os.path.exists(ckpt):
            word_model  = load_model(ckpt, use_ctc=False)
            word_labels = load_label_map(word_cache)

    if sentence_model is None:
        ckpt = os.path.join(CHECKPOINT_DIR, "best_lmfst_isl.pth")
        if os.path.exists(ckpt):
            sentence_model  = load_model(ckpt, use_ctc=False)
            sentence_labels = load_label_map(sentence_cache)

    if word_model is None and sentence_model is None:
        print("[ERROR] No trained checkpoints found. Run training first.")
        return

    # Setup 
    mode    = "word" if word_model else "sentence"
    buf     = collections.deque(maxlen=SEQUENCE_LENGTH)
    history = []
    cap     = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {WEBCAM_INDEX}")
        return

    fps_buf   = collections.deque(maxlen=30)
    label     = ""
    conf      = 0.0
    last_label = ""
    last_conf  = 0.0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ) as holistic:

        print(f"  Starting in {mode.upper()} mode…")

        while cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            h, w    = frame.shape[:2]

            #  Single MediaPipe pass
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            # Extract landmarks 
            kp = extract_frame_landmarks(frame, holistic, results=results)
            buf.append(kp)

            # Draw skeleton on BGR copy
            display = frame.copy()
            draw_landmarks(display, results)

            # Predict 
            if len(buf) == SEQUENCE_LENGTH:
                arr = np.array(buf, dtype=np.float32)   # (T, 279)
                if mode == "word" and word_model:
                    label, conf = predict(word_model, arr, word_labels)
                elif mode == "sentence" and sentence_model:
                    label, conf = predict(sentence_model, arr, sentence_labels)
                last_label = label
                last_conf  = conf

            #  FPS
            fps_buf.append(1.0 / max(time.time() - t0, 1e-6))
            fps = np.mean(fps_buf)

            # Overlay 
            # Top bar: Mode + FPS
            cv2.rectangle(display, (0, 0), (w, 44), (30, 30, 30), -1)
            cv2.putText(display,
                        f"Mode: {mode.upper()}  |  FPS: {fps:.1f}  |  Buf: {len(buf)}/{SEQUENCE_LENGTH}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2)

            # Buffer
            fill = int(w * len(buf) / SEQUENCE_LENGTH)
            cv2.rectangle(display, (0, 44), (fill, 50), (0, 180, 255), -1)
            cv2.rectangle(display, (fill, 44), (w, 50), (60, 60, 60), -1)

            # Prediction display 
            if last_label:
                # Color by confidence
                if last_conf >= PREDICTION_THRESH:
                    if last_conf > 0.80:
                        color = (0, 220, 0)        # bright green — high conf
                        status = "HIGH"
                    else:
                        color = (0, 200, 200)      # cyan — medium conf
                        status = "MED"
                else:
                    color = (150, 150, 150)        # grey — below threshold
                    status = "LOW"

                # prediction text box
                pred_h = 70
                cv2.rectangle(display, (0, h - pred_h - 45), (w, h - 45),
                               (20, 20, 20), -1)

                # Large label
                text  = f"{last_label}"
                tsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
                tx    = max(10, (w - tsize[0]) // 2)   # centered
                cv2.putText(display, text,
                            (tx, h - pred_h - 45 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.4, color, 3)

                # Confidence bar below label
                bar_y = h - 55
                bar_w = int(w * last_conf)
                cv2.rectangle(display, (0, bar_y), (w, bar_y + 10), (40, 40, 40), -1)
                cv2.rectangle(display, (0, bar_y), (bar_w, bar_y + 10), color, -1)
                cv2.putText(display,
                            f"{last_conf*100:.0f}%  [{status}]",
                            (10, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 1)

                # Sentence mode: add to history if high conf
                if mode == "sentence" and last_conf >= PREDICTION_THRESH:
                    if not history or history[-1] != last_label:
                        history.append(last_label)

            # History bar 
            hist_y = h - 44
            cv2.rectangle(display, (0, hist_y), (w, h), (20, 20, 20), -1)
            if history:
                cv2.putText(display,
                            "  →  ".join(history[-6:]),
                            (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (180, 180, 255), 2)
            else:
                cv2.putText(display,
                            "No history yet  (sentence mode only)",
                            (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (100, 100, 100), 1)

            cv2.imshow("LandmarkFusion-ST", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("m"):
                mode = "sentence" if mode == "word" else "word"
                buf.clear()
                last_label = ""
                last_conf  = 0.0
                print(f"  Switched to {mode.upper()} mode")
            elif key == ord("c"):
                history.clear()
                buf.clear()
                last_label = ""
                last_conf  = 0.0
                print("  Cleared")
            elif key == ord("s"):
                if len(buf) == SEQUENCE_LENGTH:
                    np.save(f"saved_landmarks_{int(time.time())}.npy",
                            np.array(buf))
                    print("  Saved landmarks")

    cap.release()
    cv2.destroyAllWindows()
    print("  Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--word_ckpt",     type=str, default=None)
    p.add_argument("--sentence_ckpt", type=str, default=None)
    args = p.parse_args()
    run(args.word_ckpt, args.sentence_ckpt)