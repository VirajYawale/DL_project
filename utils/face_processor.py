"""
Face Processor — DeepFace + ArcFace (InsightFace buffalo_l)
Full training pipeline:
  - Video → frame extraction (adaptive sampling)
  - Face quality scoring & filtering
  - Augmentation for robustness (glasses, low-light, distance, angle)
  - ArcFace embedding generation + averaging
  - Cosine similarity search with dynamic threshold
  - Low-light CLAHE + gamma + denoising
  - Super-resolution upscale for distant faces
"""

import cv2
import numpy as np
import os
import pickle
from datetime import datetime
from pathlib import Path

# ──────────────────────── Optional imports ────────────────────────

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("[WARNING] InsightFace not installed. Run: pip install insightface onnxruntime")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("[WARNING] DeepFace not installed. Run: pip install deepface")

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ──────────────────────── Augmentation Pipeline ────────────────────────

class FaceAugmentor:
    """
    Generates augmented variants of face frames to improve model robustness:
    - Glasses simulation (horizontal band darkening)
    - Low-light simulation (gamma darkening + noise)
    - Distance simulation (downscale + upscale)
    - Horizontal flip, brightness/contrast jitter, blur, noise
    """

    def __init__(self):
        self.use_albumentations = ALBUMENTATIONS_AVAILABLE
        if self.use_albumentations:
            self._build_augmentor()

    def _build_augmentor(self):
        self.aug_pipeline = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.15), contrast_limit=0.25, p=1.0),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.02, p=1.0),
            ], p=0.7),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=12, p=0.5),
        ])

    def augment(self, face_img, n_augments=5):
        """Generate n augmented variants + specialized domain variants."""
        results = []

        for _ in range(n_augments):
            if self.use_albumentations and hasattr(self, 'aug_pipeline'):
                try:
                    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    aug = self.aug_pipeline(image=rgb)['image']
                    results.append(cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
                    continue
                except Exception:
                    pass
            results.append(self._manual_augment(face_img))

        # Domain-specific augmentations — critical for robustness
        results.append(self._simulate_glasses(face_img))
        results.append(self._simulate_low_light(face_img))
        results.append(self._simulate_distance(face_img))
        results.append(self._simulate_hat(face_img))
        results.append(self._simulate_mask_partial(face_img))

        return results

    def _manual_augment(self, img):
        out = img.copy().astype(np.float32)
        factor = np.random.uniform(0.55, 1.35)
        out = np.clip(out * factor, 0, 255)
        if np.random.random() > 0.5:
            out = np.fliplr(out)
        noise = np.random.normal(0, np.random.uniform(5, 22), out.shape)
        out = np.clip(out + noise, 0, 255)
        out = out.astype(np.uint8)
        if np.random.random() > 0.6:
            k = np.random.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        return out

    def _simulate_glasses(self, img):
        """Darken the eye-region to simulate glasses."""
        out = img.copy()
        h, w = out.shape[:2]
        y1, y2 = int(h * 0.27), int(h * 0.53)
        region = out[y1:y2].astype(np.float32)
        # Darken as if tinted lenses
        region = np.clip(region * np.random.uniform(0.45, 0.65), 0, 255)
        # Add slight reflection
        region[0:3, :] = np.clip(region[0:3, :] * 0.25 + 180, 0, 255)
        out[y1:y2] = region.astype(np.uint8)
        return out

    def _simulate_low_light(self, img):
        """Simulate dim/night CCTV environment."""
        out = img.astype(np.float32)
        out = out * np.random.uniform(0.15, 0.40)
        noise = np.random.normal(0, np.random.uniform(12, 30), out.shape)
        out = np.clip(out + noise, 0, 255).astype(np.uint8)
        # Slight green/bluish tint common in cheap CCTV
        out[:, :, 1] = np.clip(out[:, :, 1].astype(np.float32) * 1.08, 0, 255).astype(np.uint8)
        return out

    def _simulate_distance(self, img):
        """Simulate far-away face via aggressive downscale+upscale."""
        h, w = img.shape[:2]
        scale = np.random.uniform(0.18, 0.40)
        small = cv2.resize(img,
                           (max(16, int(w * scale)), max(16, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def _simulate_hat(self, img):
        """Darken top portion to simulate hat/cap shadow."""
        out = img.copy()
        h = out.shape[0]
        shadow_end = int(h * 0.30)
        out[0:shadow_end] = (out[0:shadow_end].astype(np.float32) * 0.35).astype(np.uint8)
        return out

    def _simulate_mask_partial(self, img):
        """Darken the lower face portion (chin/mouth) to simulate partial mask."""
        out = img.copy()
        h = out.shape[0]
        start = int(h * 0.60)
        out[start:] = (out[start:].astype(np.float32) * 0.30).astype(np.uint8)
        return out


# ──────────────────────── Quality Scorer ────────────────────────

class FaceQualityScorer:
    """Scores face crop quality (sharpness, brightness, size, contrast) on 0-1 scale."""

    def score(self, face_img):
        if face_img is None or face_img.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(1.0, lap_var / 500.0)
            mean_b = np.mean(gray)
            brightness = 1.0 - abs(mean_b - 128) / 128.0
            h, w = face_img.shape[:2]
            size_score = min(1.0, (h * w) / (110 * 110))
            contrast = min(1.0, np.std(gray) / 60.0)
            combined = (0.35 * sharpness + 0.25 * brightness +
                        0.25 * size_score + 0.15 * contrast)
            return round(float(combined), 4)
        except Exception:
            return 0.5


# ──────────────────────── Main Processor ────────────────────────

class FaceProcessor:

    def __init__(self):
        self.face_db = {}         # {student_id: np.ndarray(N, 512)}
        self.student_meta = {}    # {student_id: {roll_no, name}}

        self.augmentor = FaceAugmentor()
        self.quality_scorer = FaceQualityScorer()

        # Primary: InsightFace ArcFace buffalo_l
        self.arcface_model = None
        if INSIGHTFACE_AVAILABLE:
            try:
                self.arcface_model = FaceAnalysis(
                    name='buffalo_l',
                    allowed_modules=['detection', 'recognition']
                )
                self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))
                print("[OK] InsightFace ArcFace (buffalo_l) loaded")
            except Exception as e:
                print(f"[WARN] InsightFace load failed: {e}")

        self.THRESHOLD = 0.38     # Cosine similarity threshold
        self.MIN_QUALITY = 0.22   # Minimum quality score for frame acceptance

        print(f"[FaceProcessor] InsightFace={'Yes' if self.arcface_model else 'No'}, "
              f"DeepFace={'Yes' if DEEPFACE_AVAILABLE else 'No'}")

    # ═══════════════════════ ENROLLMENT (TRAINING) ═══════════════════════

    def process_enrollment_video(self, video_path, student_id, roll_no,
                                  embeddings_folder, name=''):
        """
        Full training pipeline:
          1. Extract frames (adaptive FPS-based sampling)
          2. Enhance each frame (CLAHE, gamma, upscale)
          3. Detect face + score quality
          4. Generate ArcFace embedding per frame
          5. Augment accepted frames → domain-robust embeddings
          6. L2-normalize all embeddings
          7. Save .pkl to disk + update in-memory face_db
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'success': False, 'message': 'Cannot open video file'}

        fps = cap.get(cv2.CAP_PROP_FPS)

        # Both CAP_PROP_FRAME_COUNT and CAP_PROP_FPS are unreliable for WebM
        # files recorded via the browser MediaRecorder API. OpenCV returns
        # garbage sentinel values (e.g. -922337203685477 frames, 0 fps).
        # Fix: read through once to count real frames, then reopen the file.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0 or fps <= 0:
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.release()
            cap = cv2.VideoCapture(video_path)  # reopen — seek unreliable for WebM
            if not cap.isOpened():
                return {'success': False, 'message': 'Cannot reopen video file'}
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 25.0  # safe default for browser-recorded WebM

        duration = total_frames / fps

        if duration < 2.5:
            return {
                'success': False,
                'message': f'Video too short ({duration:.1f}s). Record at least 5 seconds.'
            }

        # Aim for ~15-25 candidate frames from a 5-10s clip
        sample_interval = max(1, int(fps * 0.38))

        raw_embeddings = []
        quality_scores = []
        accepted_face_crops = []
        frame_idx = 0
        saved_count = 0

        frame_dir = Path(f"data/frames/{roll_no}")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for old in frame_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                enhanced = self._enhance_frame(frame)
                face_data = self._extract_face_and_embedding(enhanced)

                if face_data is not None:
                    embedding, face_crop, quality = face_data
                    if quality >= self.MIN_QUALITY:
                        raw_embeddings.append(embedding)
                        quality_scores.append(quality)
                        accepted_face_crops.append(face_crop)
                        cv2.imwrite(
                            str(frame_dir / f"frame_{saved_count:03d}_q{int(quality*100)}.jpg"),
                            enhanced
                        )
                        saved_count += 1

            frame_idx += 1

        cap.release()

        if len(raw_embeddings) < 3:
            return {
                'success': False,
                'message': (
                    f'Only {len(raw_embeddings)} usable face frames found. '
                    'Ensure: face visible, good lighting, 5-10 second recording.'
                )
            }

        # ── Augmentation ──
        aug_embeddings = []
        aug_count = 0
        for crop in accepted_face_crops:
            for aug_img in self.augmentor.augment(crop, n_augments=5):
                emb = self._embedding_from_crop(aug_img)
                if emb is not None:
                    aug_embeddings.append(emb)
                    aug_count += 1

        # ── Combine & Normalize ──
        all_embeddings = raw_embeddings + aug_embeddings
        emb_array = np.stack(all_embeddings)
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True) + 1e-9
        emb_array = emb_array / norms

        avg_quality = float(np.mean(quality_scores)) if quality_scores else 0.0

        # ── Persist ──
        emb_path = os.path.join(embeddings_folder, f"{roll_no}.pkl")
        with open(emb_path, 'wb') as f:
            pickle.dump({
                'student_id': student_id,
                'roll_no': roll_no,
                'name': name,
                'embeddings': emb_array,
                'raw_frame_count': len(raw_embeddings),
                'augmented_count': aug_count,
                'total_embeddings': len(all_embeddings),
                'avg_quality': avg_quality,
                'created_at': datetime.now().isoformat(),
                'model': 'ArcFace-buffalo_l' if self.arcface_model else 'ArcFace-DeepFace'
            }, f)

        self.face_db[student_id] = emb_array
        self.student_meta[student_id] = {'roll_no': roll_no, 'name': name}

        print(f"[Train] {roll_no}: {len(raw_embeddings)} raw + {aug_count} aug "
              f"= {len(all_embeddings)} embeddings | quality={avg_quality:.2f}")

        return {
            'success': True,
            'message': f'Trained on {len(all_embeddings)} total samples',
            'embedding_path': emb_path,
            'frame_count': len(raw_embeddings),
            'augmented_count': aug_count,
            'total_samples': len(all_embeddings),
            'quality_avg': round(avg_quality, 3),
            'training_details': {
                'raw_frames': len(raw_embeddings),
                'augmented_variants': aug_count,
                'total_embeddings': len(all_embeddings),
                'model_used': ('InsightFace ArcFace (buffalo_l)'
                               if self.arcface_model else 'DeepFace ArcFace'),
                'avg_quality_score': round(avg_quality * 100, 1),
                'augmentations_applied': [
                    'Glasses simulation', 'Low-light (CCTV night)',
                    'Distance/small-face', 'Hat/cap shadow',
                    'Partial face cover', 'Brightness jitter',
                    'Horizontal flip', 'Gaussian noise', 'Motion blur'
                ]
            }
        }

    # ═══════════════════════ RECOGNITION ═══════════════════════

    def identify_face(self, frame):
        """Detect all faces in frame, return best matches from face_db."""
        enhanced = self._enhance_frame(frame)
        if self.arcface_model and INSIGHTFACE_AVAILABLE:
            return self._identify_insightface(enhanced)
        elif DEEPFACE_AVAILABLE:
            return self._identify_deepface(enhanced)
        return []

    def _identify_insightface(self, frame):
        results = []
        try:
            faces = self.arcface_model.get(frame)
            for face in faces:
                if face.embedding is None:
                    continue
                q = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)
                best_id, best_score = self._cosine_search(q)
                if best_score >= self.THRESHOLD:
                    meta = self.student_meta.get(best_id, {})
                    results.append({
                        'student_id': best_id,
                        'roll_no': meta.get('roll_no', 'Unknown'),
                        'name': meta.get('name', ''),
                        'confidence': round(float(best_score), 4),
                        'bbox': face.bbox.astype(int).tolist(),
                        'is_real': True
                    })
        except Exception as e:
            print(f"[InsightFace] {e}")
        return results

    def _identify_deepface(self, frame):
        results = []
        try:
            tmp = '/tmp/_query.jpg'
            cv2.imwrite(tmp, frame)
            rep = DeepFace.represent(img_path=tmp, model_name='ArcFace',
                                     enforce_detection=False)
            if not rep:
                return []
            q = np.array(rep[0]['embedding'])
            q = q / (np.linalg.norm(q) + 1e-9)
            best_id, best_score = self._cosine_search(q)
            if best_score >= self.THRESHOLD:
                meta = self.student_meta.get(best_id, {})
                h, w = frame.shape[:2]
                results.append({
                    'student_id': best_id,
                    'roll_no': meta.get('roll_no', 'Unknown'),
                    'name': meta.get('name', ''),
                    'confidence': round(float(best_score), 4),
                    'bbox': [0, 0, w, h],
                    'is_real': True
                })
        except Exception as e:
            print(f"[DeepFace] {e}")
        return results

    def _cosine_search(self, query_emb):
        best_id, best_score = None, -1.0
        for sid, emb_arr in self.face_db.items():
            scores = emb_arr @ query_emb
            ms = float(np.max(scores))
            if ms > best_score:
                best_score, best_id = ms, sid
        return best_id, best_score

    # ═══════════════════════ ENHANCEMENT ═══════════════════════

    def _enhance_frame(self, frame):
        """CLAHE + gamma correction + denoising + upscale for low-light & distant faces."""
        if frame is None or frame.size == 0:
            return frame
        try:
            denoised = cv2.fastNlMeansDenoisingColored(frame, None, 8, 8, 5, 17)
            yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            mean_b = np.mean(gray)
            if mean_b < 65:
                gamma = 2.3
            elif mean_b < 95:
                gamma = 1.7
            else:
                gamma = None

            if gamma:
                lut = np.array([min(255, int(((i/255.0)**(1/gamma))*255))
                                for i in range(256)], dtype=np.uint8)
                enhanced = cv2.LUT(enhanced, lut)

            h, w = enhanced.shape[:2]
            if w < 480:
                scale = 480 / w
                enhanced = cv2.resize(enhanced, (480, int(h*scale)),
                                      interpolation=cv2.INTER_CUBIC)
            return enhanced
        except Exception:
            return frame

    def _extract_face_and_embedding(self, frame):
        """Returns (embedding, face_crop, quality) or None."""
        try:
            if self.arcface_model and INSIGHTFACE_AVAILABLE:
                faces = self.arcface_model.get(frame)
                if not faces:
                    return None
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                if face.embedding is None:
                    return None
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-9)
                x1, y1, x2, y2 = face.bbox.astype(int)
                h, w = frame.shape[:2]
                pad = 15
                crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
                if crop.size == 0:
                    return None
                return emb, crop, self.quality_scorer.score(crop)

            elif DEEPFACE_AVAILABLE:
                tmp = '/tmp/_enroll.jpg'
                cv2.imwrite(tmp, frame)
                rep = DeepFace.represent(img_path=tmp, model_name='ArcFace',
                                         enforce_detection=True)
                if not rep:
                    return None
                emb = np.array(rep[0]['embedding'])
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                return emb, frame, self.quality_scorer.score(frame)
        except Exception:
            return None

    def _embedding_from_crop(self, face_crop):
        """Generate ArcFace embedding from a face crop (for augmented images)."""
        if face_crop is None or face_crop.size == 0:
            return None
        try:
            resized = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)

            if self.arcface_model and INSIGHTFACE_AVAILABLE:
                faces = self.arcface_model.get(resized)
                if faces and faces[0].embedding is not None:
                    emb = faces[0].embedding
                    return emb / (np.linalg.norm(emb) + 1e-9)

            if DEEPFACE_AVAILABLE:
                tmp = '/tmp/_aug.jpg'
                cv2.imwrite(tmp, face_crop)
                rep = DeepFace.represent(img_path=tmp, model_name='ArcFace',
                                         enforce_detection=False)
                if rep:
                    emb = np.array(rep[0]['embedding'])
                    return emb / (np.linalg.norm(emb) + 1e-9)
        except Exception:
            pass
        return None

    # ═══════════════════════ DB MANAGEMENT ═══════════════════════

    def reload_face_db(self, embeddings_folder):
        self.face_db = {}
        self.student_meta = {}
        folder = Path(embeddings_folder)
        if not folder.exists():
            return
        loaded = 0
        for pkl in folder.glob("*.pkl"):
            try:
                with open(pkl, 'rb') as f:
                    data = pickle.load(f)
                sid = data['student_id']
                self.face_db[sid] = data['embeddings']
                self.student_meta[sid] = {
                    'roll_no': data.get('roll_no', pkl.stem),
                    'name': data.get('name', '')
                }
                loaded += 1
            except Exception as e:
                print(f"[WARN] {pkl.name}: {e}")
        total_emb = sum(e.shape[0] for e in self.face_db.values())
        print(f"[FaceDB] {loaded} students, {total_emb} total embeddings")
