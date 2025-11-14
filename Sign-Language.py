"""
Requirements:
  pip install opencv-python mediapipe numpy
  pip install scikit-learn joblib
  pip install gTTS
  pip install playsound==1.2.2

How it works:
  - Press a letter key (a..z) to capture a sample for that letter (one hand only).
  - Press '1' to save dataset to features.npy and labels.npy.
  - Press '2' to train a KNN classifier from current dataset.
  - Press '3' to run real-time recognition using the trained model.
  - Press '4' to go back to collect mode.
  - Press '5' to speak the current word each time.
  - Press '6' to clear the current word.
  - Press '7' to start quiz mode.
  - Press ESC to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import string
import os
import time
import threading
from gtts import gTTS
from playsound import playsound
import random
import math

try:
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATASET_FEAT = 'features.npy'
DATASET_LABEL = 'labels.npy'
MODEL_FILE = 'knn_model.joblib'

def lm_to_vector(landmarks, image_w, image_h):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = coords[0].copy()
    coords = coords - wrist
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs
    return coords.flatten()

def compute_similarity_score(feat, X, y, predicted_label):
    X = np.array(X)
    y = np.array(y)
    same_label_samples = X[y == predicted_label]
    if len(same_label_samples) == 0:
        return 0.0
    distances = np.linalg.norm(same_label_samples - feat, axis=1)
    avg_dist = np.mean(distances)
    scaled = max(0.0, 1 - avg_dist)
    return scaled * 100.0

class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict_single(self, x):
        dists = np.linalg.norm(self.X - x, axis=1)
        idx = np.argsort(dists)[: self.k]
        votes = self.y[idx]
        values, counts = np.unique(votes, return_counts=True)
        confidence = 1 - (np.min(dists) / np.max(dists)) if np.max(dists) > 0 else 1.0
        return values[np.argmax(counts)], confidence

def load_dataset():
    if os.path.exists(DATASET_FEAT) and os.path.exists(DATASET_LABEL):
        X = np.load(DATASET_FEAT)
        y = np.load(DATASET_LABEL)
        print(f"Loaded dataset: {len(X)} samples")
        return list(X), list(y)
    return [], []

def save_dataset(X, y):
    np.save(DATASET_FEAT, np.array(X))
    np.save(DATASET_LABEL, np.array(y))
    print(f"Saved dataset: {len(X)} samples")

def train_model(X, y, k=5):
    if len(X) == 0:
        print("No data to train on. Capture samples first.")
        return None
    if SKLEARN_AVAILABLE:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X, y)
        joblib.dump(clf, MODEL_FILE)
        print("Trained sklearn KNN and saved.")
        return clf
    else:
        clf = SimpleKNN(k=k)
        clf.fit(X, y)
        print("Trained SimpleKNN")
        return clf

def load_model():
    if SKLEARN_AVAILABLE and os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None

# ---- TTS helpers (gTTS + playsound in background thread) ----
def _tts_play(text, filename):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filename)
        playsound(filename)
    except Exception as e:
        print("TTS Error:", e)
    finally:
        try:
            if os.path.exists(filename):
                os.remove(filename)
        except Exception:
            pass

def speak_text_async(text, prefix="tts"):
    if not text:
        return
    fname = f"{prefix}_{int(time.time() * 1000)}.mp3"
    threading.Thread(target=_tts_play, args=(text, fname), daemon=True).start()

def speak_word_thread(word):
    speak_text_async(word, prefix="word")

def speak_feedback_correct():
    speak_text_async("Correct", prefix="fb_correct")

def speak_feedback_wrong():
    speak_text_async("Try again", prefix="fb_wrong")

# ---------------- Main ----------------
def main():
    X, y = load_dataset()
    clf = load_model()

    LETTER_KEYS = {ord(c): c.upper() for c in string.ascii_lowercase}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    word = ""
    last_letter = None
    start_time = None
    hold_seconds = 3  # hold duration

    # Quiz variables
    quiz_target = None
    in_quiz = False
    quiz_collecting = False
    quiz_start_time = None
    quiz_collected_preds = []
    quiz_pause_until = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:

        mode = 'collect'
        print("Controls: a..z capture, 1 save, 2 train, 3 run, 4 collect, 5 speak, 6 clear, 7 quiz, ESC quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            img = cv2.flip(frame, 1)
            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            display_text = ""
            if mode == 'collect':
                display_text = f"Mode: {mode} | Samples: {len(X)}"

            trained_letters = sorted(list(set(y))) if len(y) > 0 else list(string.ascii_uppercase)

            if in_quiz and (quiz_target is None) and len(trained_letters) > 0:
                quiz_target = random.choice(trained_letters)
                quiz_collecting = False
                quiz_start_time = None
                quiz_collected_preds = []
                quiz_pause_until = 0

            hand_detected = False

            if res.multi_hand_landmarks:
                hand_detected = True
                lm = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x_min, x_max = min(xs) - 15, max(xs) + 15
                y_min, y_max = min(ys) - 15, max(ys) + 15
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                feat = lm_to_vector(lm.landmark, w, h)

                # Run mode
                if mode == "run" and clf is not None and not in_quiz:
                    if not SKLEARN_AVAILABLE:
                        pred, conf = clf.predict_single(feat)
                        percentage = conf * 100
                    else:
                        pred = clf.predict([feat])[0]
                        percentage = compute_similarity_score(feat, X, np.array(y), pred)

                    if pred == last_letter:
                        if start_time is None:
                            start_time = time.time()
                        held = time.time() - start_time
                        cv2.putText(img, f"Holding {pred}: {held:.1f}s",
                                    (w//2 - 150, h//2 + 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                        if held >= hold_seconds:
                            word += pred
                            start_time = None
                            last_letter = None
                    else:
                        last_letter = pred
                        start_time = None

                    cv2.putText(img, pred, (w//2 - 40, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.6, (0,255,0), 6)
                    cv2.putText(img, f"{percentage:.2f}%", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

                # Quiz mode
                elif in_quiz and clf is not None:
                    if time.time() < quiz_pause_until:
                        pass
                    else:
                        if hand_detected:
                            if not quiz_collecting:
                                quiz_collecting = True
                                quiz_start_time = time.time()
                                quiz_collected_preds = []

                            if not SKLEARN_AVAILABLE:
                                pred_frame, _ = clf.predict_single(feat)
                            else:
                                pred_frame = clf.predict([feat])[0]

                            quiz_collected_preds.append(pred_frame)
                            elapsed = time.time() - quiz_start_time
                            remaining = hold_seconds - elapsed
                            if remaining > 0:
                                count_num = int(math.ceil(remaining))
                                cv2.putText(img, str(count_num), (w//2 - 20, h//2 + 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,255,255), 6)
                            else:
                                most_common = max(set(quiz_collected_preds), key=quiz_collected_preds.count) if quiz_collected_preds else None
                                if most_common == quiz_target:
                                    speak_feedback_correct()
                                    quiz_pause_until = time.time() + 1.0
                                    quiz_target = None
                                else:
                                    speak_feedback_wrong()
                                    quiz_pause_until = time.time() + 1.0
                                quiz_collecting = False
                                quiz_start_time = None
                                quiz_collected_preds = []
                                last_letter = None
                                start_time = None

            if display_text:
                cv2.putText(img, display_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if in_quiz:
                header = "QUIZ MODE - Show this letter"
                cv2.putText(img, header, (w//2 - 260, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
                if quiz_target is not None:
                    cv2.putText(img, quiz_target, (w//2 - 40, h//2 - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0,255,0), 8)
            else:
                if mode == 'run':
                    cv2.putText(img, f"Word: {word}", (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            img_big = cv2.resize(img, (900, 640))
            cv2.imshow("ASL A-Z", img_big)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            # Controls
            if key == ord("5") and len(word) > 0:
                speak_word_thread(word)
            elif key == ord("6"):
                word = ""
                print("Word cleared.")
            elif key == ord("1"):
                save_dataset(X, y)
            elif key == ord("2"):
                clf = train_model(X, y)
            elif key == ord("3"):
                if clf is None:
                    print("Train first.")
                else:
                    mode = "run"
                    in_quiz = False
                    quiz_target = None
                    quiz_collecting = False
                    quiz_start_time = None
                    quiz_collected_preds = []
                    quiz_pause_until = 0
                    last_letter = None
                    start_time = None
            elif key == ord("4"):
                mode = "collect"
                in_quiz = False
                quiz_target = None
                quiz_collecting = False
                quiz_start_time = None
                quiz_collected_preds = []
                quiz_pause_until = 0
                last_letter = None
                start_time = None
            elif key == ord("7"):
                if clf is None:
                    print("Train first.")
                else:
                    mode = "run"
                    in_quiz = True
                    quiz_target = None
                    quiz_collecting = False
                    quiz_start_time = None
                    quiz_collected_preds = []
                    quiz_pause_until = 0
                    last_letter = None
                    start_time = None
                    trained_letters = sorted(list(set(y))) if len(y) > 0 else list(string.ascii_uppercase)
                    if len(trained_letters) == 0:
                        print("No trained letters available for quiz.")
                        in_quiz = False
            elif key in LETTER_KEYS and mode == "collect":
                if res.multi_hand_landmarks:
                    feat = lm_to_vector(res.multi_hand_landmarks[0].landmark, w, h)
                    X.append(feat)
                    y.append(LETTER_KEYS[key])
                    print(f"Captured '{LETTER_KEYS[key]}'")
                else:
                    print("No hand detected.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
