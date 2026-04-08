"""
Smart Attendance System — Flask Application
DeepFace + ArcFace (InsightFace buffalo_l) | Anti-Spoofing | Model Training Pipeline
"""

from flask import Flask, render_template, request, jsonify, Response, session
from flask_sqlalchemy import SQLAlchemy
import cv2
import os
import json
import base64
import numpy as np
from datetime import datetime, date
import threading
import time

from database.models import db, Student, Subject, Enrollment, Attendance, FaceEmbedding
from utils.face_processor import FaceProcessor
from utils.anti_spoof import AntiSpoofDetector
from utils.attendance_engine import AttendanceEngine

app = Flask(__name__)
app.secret_key = "smart_attendance_2024"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'data/videos'
app.config['EMBEDDINGS_FOLDER'] = 'data/embeddings'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['EMBEDDINGS_FOLDER'], exist_ok=True)
os.makedirs('data/frames', exist_ok=True)

db.init_app(app)

# ── Global instances ──
face_processor = FaceProcessor()
anti_spoof = AntiSpoofDetector()
attendance_engine = AttendanceEngine(face_processor, anti_spoof)

live_stream_active = False
current_subject_id = None

# ── Camera stream globals (smooth MJPEG feed) ──
_camera      = None
_camera_lock = threading.Lock()
_stream_on   = False   # True while attendance session is active

with app.app_context():
    db.create_all()
    # Pre-load embeddings if they exist
    face_processor.reload_face_db(app.config['EMBEDDINGS_FOLDER'])


# ═══════════════════════════ MJPEG CAMERA STREAM ═══════════════════════════

def _get_camera():
    global _camera
    if _camera is None or not _camera.isOpened():
        _camera = cv2.VideoCapture(0)
        _camera.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        _camera.set(cv2.CAP_PROP_FPS, 30)
    return _camera


def _release_camera():
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


def _gen_frames():
    """
    Server-side MJPEG generator.
    Each frame is: liveness update → face detect + spoof check → annotate → encode.
    Smooth because it's a continuous server push (no client JS polling overhead).
    """
    cam = _get_camera()
    while _stream_on:
        with _camera_lock:
            ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue

        # ── Liveness gate: update blink state every frame ──
        anti_spoof.update_liveness(frame)
        liveness_ok = anti_spoof.liveness_passed()

        blink_count = anti_spoof.liveness_status()["blink_count"]
        required    = anti_spoof.liveness_status()["required"]

        if not liveness_ok:
            # Draw liveness overlay on raw frame
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            msg = f"Blink {blink_count}/{required}x to verify liveness"
            cv2.putText(frame, msg, (12, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)
        else:
            # ── Run attendance pipeline on this frame ──
            if current_subject_id:
                result = attendance_engine.process_frame(frame, current_subject_id)
                frame  = attendance_engine.draw_annotations(frame, result)

            # Liveness confirmed badge
            cv2.rectangle(frame, (0, 0), (220, 32), (0, 150, 80), -1)
            cv2.putText(frame, "  LIVENESS OK", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Encode JPEG
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Smooth MJPEG stream endpoint consumed by <img> tag in attendance.html."""
    return Response(_gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/liveness/status')
def liveness_status_route():
    return jsonify(anti_spoof.liveness_status())


@app.route('/api/liveness/reset', methods=['POST'])
def liveness_reset_route():
    anti_spoof.reset_liveness()
    return jsonify({'message': 'Liveness reset'})


# ═══════════════════════════ DASHBOARD ═══════════════════════════

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


# ═══════════════════════════ STUDENT MANAGEMENT ═══════════════════════════

@app.route('/students')
def students():
    return render_template('students.html')

@app.route('/api/students', methods=['GET'])
def get_students():
    return jsonify([s.to_dict() for s in Student.query.all()])

@app.route('/api/students', methods=['POST'])
def add_student():
    data = request.json
    if Student.query.filter_by(roll_no=data['roll_no']).first():
        return jsonify({'error': 'Roll number already exists'}), 400
    student = Student(
        roll_no=data['roll_no'], name=data['name'],
        section=data['section'], email=data.get('email', ''),
        phone=data.get('phone', '')
    )
    db.session.add(student)
    db.session.commit()
    return jsonify(student.to_dict()), 201

@app.route('/api/students/<int:sid>', methods=['DELETE'])
def delete_student(sid):
    student = Student.query.get_or_404(sid)
    db.session.delete(student)
    db.session.commit()
    return jsonify({'message': 'Deleted'})


# ═══════════════════════════ SUBJECT MANAGEMENT ═══════════════════════════

@app.route('/subjects')
def subjects():
    return render_template('subjects.html')

@app.route('/api/subjects', methods=['GET'])
def get_subjects():
    return jsonify([s.to_dict() for s in Subject.query.all()])

@app.route('/api/subjects', methods=['POST'])
def add_subject():
    data = request.json
    subject = Subject(
        code=data['code'], name=data['name'],
        faculty=data.get('faculty', ''), section=data.get('section', '')
    )
    db.session.add(subject)
    db.session.commit()
    return jsonify(subject.to_dict()), 201


# ═══════════════════════════ ENROLLMENT + MODEL TRAINING ═══════════════════════════

@app.route('/enroll')
def enroll():
    students_list = Student.query.all()
    subjects_list = Subject.query.all()
    return render_template('enroll.html', students=students_list, subjects=subjects_list)


@app.route('/api/enroll', methods=['POST'])
def enroll_student():
    """
    Full enrollment + training pipeline:
      1. Decode base64 video → save to disk
      2. face_processor.process_enrollment_video():
         - Frame extraction (adaptive sampling)
         - Frame enhancement (CLAHE, gamma, denoising, upscale)
         - Face detection + quality filtering
         - ArcFace embedding generation (InsightFace buffalo_l)
         - Augmentation (glasses, low-light, distance, hat, flip, noise, blur)
         - L2 normalize all embeddings
         - Save .pkl to embeddings_folder
      3. Update DB (FaceEmbedding record + Enrollment records)
      4. Reload in-memory face DB for immediate recognition
    """
    data = request.json
    student_id  = data.get('student_id')
    subject_ids = data.get('subject_ids', [])
    video_b64   = data.get('video_b64')

    if not student_id or not video_b64:
        return jsonify({'error': 'student_id and video_b64 are required'}), 400

    student = Student.query.get_or_404(student_id)

    # ── Decode & save video ──
    try:
        video_bytes = base64.b64decode(video_b64)
    except Exception:
        return jsonify({'error': 'Invalid base64 video data'}), 400

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    video_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        f"{student.roll_no}_{timestamp}.webm"
    )
    with open(video_path, 'wb') as f:
        f.write(video_bytes)

    # ── Train the model ──
    result = face_processor.process_enrollment_video(
        video_path=video_path,
        student_id=student_id,
        roll_no=student.roll_no,
        name=student.name,
        embeddings_folder=app.config['EMBEDDINGS_FOLDER']
    )

    if not result['success']:
        return jsonify({'error': result['message']}), 400

    # ── Update DB ──
    emb = FaceEmbedding.query.filter_by(student_id=student_id).first()
    if emb:
        emb.embedding_path = result['embedding_path']
        emb.frame_count    = result['total_samples']
        emb.updated_at     = datetime.utcnow()
    else:
        emb = FaceEmbedding(
            student_id=student_id,
            embedding_path=result['embedding_path'],
            frame_count=result['total_samples']
        )
        db.session.add(emb)

    for sid in subject_ids:
        if not Enrollment.query.filter_by(student_id=student_id, subject_id=sid).first():
            db.session.add(Enrollment(student_id=student_id, subject_id=sid))

    db.session.commit()

    # ── Reload face DB ──
    attendance_engine.reload_face_db(app.config['EMBEDDINGS_FOLDER'])

    return jsonify({
        'message':          f"Model trained & {student.name} enrolled successfully",
        'frames_used':      result['frame_count'],
        'augmented_count':  result['augmented_count'],
        'total_samples':    result['total_samples'],
        'quality_avg':      result['quality_avg'],
        'subjects_enrolled': len(subject_ids),
        'training_details': result['training_details']
    })


@app.route('/api/enroll/status/<int:student_id>', methods=['GET'])
def enrollment_status(student_id):
    """Return training stats for a student."""
    emb = FaceEmbedding.query.filter_by(student_id=student_id).first()
    if not emb:
        return jsonify({'enrolled': False})
    stats = face_processor.get_enrollment_stats(student_id)
    return jsonify({
        'enrolled': True,
        'frame_count': emb.frame_count,
        'embedding_path': emb.embedding_path,
        'updated_at': emb.updated_at.isoformat() if emb.updated_at else None,
        'in_memory': stats is not None,
        'total_embeddings': stats['total_embeddings'] if stats else emb.frame_count
    })


# ═══════════════════════════ ATTENDANCE (LIVE) ═══════════════════════════

@app.route('/attendance')
def attendance():
    return render_template('attendance.html', subjects=Subject.query.all())

@app.route('/api/attendance/start', methods=['POST'])
def start_attendance():
    global live_stream_active, current_subject_id, _stream_on
    data = request.json
    current_subject_id = data['subject_id']
    live_stream_active = True
    _stream_on = True
    anti_spoof.reset_liveness()          # fresh liveness check each session
    attendance_engine.start_session(current_subject_id)
    return jsonify({'message': 'Session started'})

@app.route('/api/attendance/stop', methods=['POST'])
def stop_attendance():
    global live_stream_active, _stream_on
    live_stream_active = False
    _stream_on = False
    summary = attendance_engine.stop_session()
    _release_camera()
    return jsonify({'message': 'Session stopped', 'summary': summary})

@app.route('/api/attendance/frame', methods=['POST'])
def process_frame():
    data = request.json
    frame_b64 = data['frame']
    frame_bytes = base64.b64decode(frame_b64.split(',')[-1])
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    result = attendance_engine.process_frame(frame, current_subject_id)
    return jsonify(result)

@app.route('/api/attendance/log', methods=['GET'])
def get_attendance_log():
    subject_id = request.args.get('subject_id')
    date_str   = request.args.get('date', str(date.today()))
    query_date = datetime.strptime(date_str, '%Y-%m-%d').date()

    records = (db.session.query(Attendance, Student, Subject)
               .join(Student, Attendance.student_id == Student.id)
               .join(Subject, Attendance.subject_id == Subject.id)
               .filter(Attendance.subject_id == subject_id)
               .filter(db.func.date(Attendance.timestamp) == query_date)
               .all())

    return jsonify([{
        'roll_no':    s.roll_no,
        'name':       s.name,
        'section':    s.section,
        'subject':    sub.name,
        'time':       a.timestamp.strftime('%H:%M:%S'),
        'confidence': a.confidence,
        'method':     a.method
    } for a, s, sub in records])

@app.route('/api/attendance/mark', methods=['POST'])
def mark_attendance():
    data = request.json
    existing = Attendance.query.filter_by(
        student_id=data['student_id'],
        subject_id=data['subject_id'],
        date=date.today()
    ).first()
    if existing:
        return jsonify({'message': 'Already marked'})
    record = Attendance(
        student_id=data['student_id'],
        subject_id=data['subject_id'],
        confidence=data['confidence'],
        method=data.get('method', 'ArcFace'),
        date=date.today()
    )
    db.session.add(record)
    db.session.commit()
    return jsonify({'message': 'Marked'})

@app.route('/api/reports')
def get_reports():
    subject_id = request.args.get('subject_id')
    students = (db.session.query(Student)
                .join(Enrollment, Student.id == Enrollment.student_id)
                .filter(Enrollment.subject_id == subject_id)
                .all())
    report = []
    for student in students:
        total = Attendance.query.filter_by(
            student_id=student.id, subject_id=subject_id).count()
        report.append({
            'roll_no': student.roll_no, 'name': student.name,
            'section': student.section, 'total_present': total
        })
    return jsonify(report)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
