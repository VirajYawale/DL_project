"""
Attendance Engine
Coordinates: frame → face detection → anti-spoof → recognition → DB mark
Includes debounce/cooldown so same student isn't marked multiple times.
"""

import cv2
import numpy as np
import requests
from datetime import datetime


class AttendanceEngine:

    def __init__(self, face_processor, anti_spoof_detector):
        self.fp = face_processor
        self.spoof = anti_spoof_detector

        self.session_active = False
        self.current_subject_id = None
        self.marked_today = set()        # {student_id}
        self.last_seen = {}              # {student_id: datetime}
        self.COOLDOWN_SEC = 30
        self.session_log = []

        self.MIN_FACE_CONFIDENCE = 0.38
        self.MIN_SPOOF_SCORE = 0.30

    def start_session(self, subject_id):
        self.session_active = True
        self.current_subject_id = subject_id
        self.marked_today = set()
        self.last_seen = {}
        self.session_log = []
        print(f"[Session] Started for subject {subject_id}")

    def stop_session(self):
        self.session_active = False
        summary = {
            'subject_id': self.current_subject_id,
            'marked_count': len(self.marked_today),
            'log': self.session_log
        }
        self.current_subject_id = None
        print(f"[Session] Stopped — marked {summary['marked_count']} students")
        return summary

    def reload_face_db(self, embeddings_folder):
        self.fp.reload_face_db(embeddings_folder)

    def process_frame(self, frame, subject_id):
        if frame is None:
            return {'faces': [], 'error': 'Empty frame'}

        results = []
        detections = self.fp.identify_face(frame)

        for det in detections:
            student_id = det['student_id']
            confidence = det['confidence']
            bbox = det['bbox']

            if confidence < self.MIN_FACE_CONFIDENCE:
                results.append({
                    'label': 'Unknown',
                    'name': '',
                    'confidence': confidence,
                    'bbox': bbox,
                    'status': 'low_confidence',
                    'is_spoof': False
                })
                continue

            # Anti-spoof check
            face_crop = self.spoof.extract_face_crop(frame, bbox)
            is_real, spoof_score, spoof_method = self.spoof.check(face_crop, frame)

            if not is_real or spoof_score < self.MIN_SPOOF_SCORE:
                results.append({
                    'label': det['roll_no'],
                    'name': det.get('name', ''),
                    'confidence': confidence,
                    'bbox': bbox,
                    'status': 'spoof_detected',
                    'is_spoof': True,
                    'spoof_score': spoof_score,
                    'method': spoof_method
                })
                continue

            # Debounce
            now = datetime.now()
            last = self.last_seen.get(student_id)
            already_marked = student_id in self.marked_today
            on_cooldown = last and (now - last).seconds < self.COOLDOWN_SEC

            if not already_marked and not on_cooldown and self.session_active:
                self._mark_attendance(student_id, subject_id, confidence, spoof_score)
                self.marked_today.add(student_id)
                self.last_seen[student_id] = now
                self.session_log.append({
                    'student_id': student_id,
                    'roll_no': det['roll_no'],
                    'name': det.get('name', ''),
                    'time': now.strftime('%H:%M:%S'),
                    'confidence': confidence
                })
                status = 'marked'
            elif already_marked:
                status = 'already_marked'
            else:
                status = 'cooldown'

            results.append({
                'label': det['roll_no'],
                'name': det.get('name', ''),
                'confidence': confidence,
                'bbox': bbox,
                'status': status,
                'is_spoof': False,
                'spoof_score': spoof_score,
                'method': spoof_method
            })

        return {
            'faces': results,
            'total_marked': len(self.marked_today),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }

    def _mark_attendance(self, student_id, subject_id, confidence, spoof_score):
        try:
            requests.post('http://localhost:5000/api/attendance/mark', json={
                'student_id': student_id,
                'subject_id': subject_id,
                'confidence': confidence,
                'spoof_score': spoof_score,
                'method': 'ArcFace'
            }, timeout=2)
        except Exception as e:
            print(f"[AttendanceEngine] DB mark failed: {e}")

    def draw_annotations(self, frame, results):
        for face in results.get('faces', []):
            bbox = face['bbox']
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox

            if face['is_spoof']:
                color = (0, 0, 255)
                label = f"SPOOF: {face['label']}"
            elif face['status'] == 'marked':
                color = (0, 255, 0)
                label = f"{face['label']} {face['name']} ({face['confidence']:.2f})"
            elif face['status'] == 'already_marked':
                color = (255, 165, 0)
                label = f"{face['label']} ✓"
            else:
                color = (100, 100, 100)
                label = face['label']

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
        cv2.putText(frame, f"Marked: {results.get('total_marked', 0)}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame
