"""
Anti-Spoofing Detector
Uses Silent-Face (MiniFASNet) if models are present, else falls back to
classical texture analysis (LBP + FFT + gradient).
Detects: photo printouts, phone/tablet screens, video replay attacks.
"""

import cv2
import numpy as np
import os
import sys

# Absolute path so model paths resolve correctly regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)   # project root


class TextureSpoofDetector:
    """
    Classical anti-spoof using LBP texture + FFT frequency + gradient analysis.
    No deep model required — reliable fallback.
    """

    def __init__(self):
        self.threshold = 0.33

    def predict(self, face_crop):
        """Returns (is_real: bool, score: float 0-1)"""
        if face_crop is None or face_crop.size == 0:
            return True, 0.5
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 128))
            lbp  = self._lbp_score(gray)
            fft  = self._fft_score(gray)
            grad = self._gradient_score(gray)
            score = 0.40 * lbp + 0.35 * fft + 0.25 * grad
            return score > self.threshold, round(float(score), 4)
        except Exception as e:
            print(f"[AntiSpoof texture] {e}")
            return True, 0.5

    def _lbp_score(self, gray):
        lbp = np.zeros_like(gray)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp += (gray >= shifted).astype(np.uint8)
        hist, _ = np.histogram(lbp, bins=8, range=(0, 8), density=True)
        uniformity = 1.0 - np.std(hist) * 4
        return max(0.0, min(1.0, uniformity))

    def _fft_score(self, gray):
        f      = np.fft.fft2(gray.astype(float))
        fshift = np.fft.fftshift(f)
        mag    = np.abs(fshift)
        h, w   = mag.shape
        cy, cx = h // 2, w // 2
        r      = min(h, w) // 4
        y, x   = np.ogrid[:h, :w]
        inner  = (y - cy)**2 + (x - cx)**2 < (r // 2)**2
        outer  = (y - cy)**2 + (x - cx)**2 < r**2
        ring   = outer & ~inner
        mid    = np.sum(mag[ring])
        total  = np.sum(mag) + 1e-9
        ratio  = mid / total
        score  = 1.0 - abs(ratio - 0.3) * 3
        return max(0.0, min(1.0, score))

    def _gradient_score(self, gray):
        gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        var = np.var(np.sqrt(gx**2 + gy**2))
        return min(1.0, var / 2000.0)


class SilentFaceDetector:
    """
    Wrapper for Silent-Face-Anti-Spoofing (MiniFASNet).
    Looks for models at <project>/models/anti_spoof_models/
    and src at <project>/models/silent_face/src/
    Falls back gracefully if not found.
    """

    def __init__(self):
        self.available   = False
        self.model_dir   = os.path.join(_PROJ, 'models', 'anti_spoof_models')
        self._sf_src     = os.path.join(_PROJ, 'models', 'silent_face')

        if not (os.path.isdir(self.model_dir) and os.listdir(self.model_dir)):
            print("[AntiSpoof] Silent-Face model dir not found — using texture fallback")
            return

        try:
            # Insert silent_face root so `from src.xxx` imports work
            if self._sf_src not in sys.path:
                sys.path.insert(0, self._sf_src)

            from src.anti_spoof_predict import AntiSpoofPredict
            from src.generate_patches import CropImage

            self.model         = AntiSpoofPredict(0)
            self.image_cropper = CropImage()
            self.available     = True
            print("[OK] Silent-Face (MiniFASNet) anti-spoofing loaded")
        except Exception as e:
            print(f"[WARN] Silent-Face load failed: {e} — using texture fallback")

    def predict(self, face_crop, full_frame=None):
        if not self.available or full_frame is None:
            return None
        try:
            return self._run_model(full_frame)
        except Exception as e:
            print(f"[SilentFace predict] {e}")
            return None

    def _run_model(self, frame):
        image_bbox = [0, 0, frame.shape[1], frame.shape[0]]
        prediction = np.zeros((1, 3))

        for model_name in os.listdir(self.model_dir):
            if not model_name.endswith('.pth'):
                continue
            model_path = os.path.join(self.model_dir, model_name)
            name_lower = model_name.lower()
            if '80x80' in name_lower:
                h_in, w_in, scale = 80, 80, 2.7
            elif '160x160' in name_lower:
                h_in, w_in, scale = 160, 160, 4.0
            else:
                continue
            param = {
                "org_img": frame, "bbox": image_bbox,
                "scale": scale, "out_w": w_in, "out_h": h_in, "crop": True
            }
            img = self.image_cropper.crop(**param)
            prediction += self.model.predict(img, model_path)

        label = int(np.argmax(prediction))
        score = float(prediction[0][label] / (np.sum(prediction) + 1e-9))
        return label == 1, score   # label==1 → real face


class BlinkLivenessDetector:
    """
    Liveness via blink detection using OpenCV eye cascade — no dlib needed.
    Eyes visible = open; no eyes inside face ROI = closed (blink).
    Requires N blinks within a time window to pass liveness.
    Ported from Smart_Attendance_System_v3.
    """

    BLINK_REQUIRED    = 2    # blinks needed to pass
    EAR_CONSEC_FRAMES = 2    # consecutive eye-closed frames = one blink event
    WINDOW_SECONDS    = 30   # reset if no blink activity within this window

    def __init__(self):
        import time
        self._time = time
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        self._state = {
            "blink_count":  0,
            "consec_below": 0,
            "passed":       False,
            "last_reset":   self._time.time()
        }

    def reset(self):
        self._state.update({
            "blink_count":  0,
            "consec_below": 0,
            "passed":       False,
            "last_reset":   self._time.time()
        })

    @property
    def passed(self):
        return self._state["passed"]

    @property
    def blink_count(self):
        return self._state["blink_count"]

    def update(self, frame_bgr):
        """
        Call once per frame. Returns True if liveness is confirmed.
        Updates internal blink state using eye cascade inside face regions.
        """
        if self._state["passed"]:
            return True

        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        for (fx, fy, fw, fh) in faces:
            roi  = gray[fy:fy + fh, fx:fx + fw]
            eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 10, minSize=(20, 20))

            if len(eyes) == 0:
                # No eyes detected inside face → eyes likely closed
                self._state["consec_below"] += 1
            else:
                if self._state["consec_below"] >= self.EAR_CONSEC_FRAMES:
                    self._state["blink_count"] += 1
                self._state["consec_below"] = 0

            if self._state["blink_count"] >= self.BLINK_REQUIRED:
                self._state["passed"] = True
                return True

        return False

    def status(self):
        return {
            "blink_count": self._state["blink_count"],
            "required":    self.BLINK_REQUIRED,
            "passed":      self._state["passed"]
        }


class AntiSpoofDetector:
    """Main interface. Prefers SilentFace, falls back to texture analysis.
    Also includes BlinkLivenessDetector for camera-stream liveness gating."""

    def __init__(self):
        self.silent_face = SilentFaceDetector()
        self.texture     = TextureSpoofDetector()
        self.liveness    = BlinkLivenessDetector()
        method = "SilentFace (MiniFASNet)" if self.silent_face.available else "TextureFallback"
        print(f"[AntiSpoof] Active method: {method}")
        print(f"[AntiSpoof] Blink liveness: enabled (requires {BlinkLivenessDetector.BLINK_REQUIRED} blinks)")

    def check(self, face_crop, full_frame=None):
        """Returns (is_real: bool, spoof_score: float, method: str)"""
        if self.silent_face.available and full_frame is not None:
            result = self.silent_face.predict(face_crop, full_frame)
            if result is not None:
                return result[0], result[1], 'SilentFace'
        is_real, score = self.texture.predict(face_crop)
        return is_real, score, 'TextureAnalysis'

    def update_liveness(self, frame_bgr):
        """Update blink liveness state. Call once per streamed frame."""
        return self.liveness.update(frame_bgr)

    def liveness_passed(self):
        return self.liveness.passed

    def liveness_status(self):
        return self.liveness.status()

    def reset_liveness(self):
        self.liveness.reset()

    def extract_face_crop(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        pad  = 20
        return frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]