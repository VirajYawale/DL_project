"""
Database Models for Smart Attendance System
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date

db = SQLAlchemy()


class Student(db.Model):
    __tablename__ = 'students'
    id         = db.Column(db.Integer, primary_key=True)
    roll_no    = db.Column(db.String(20), unique=True, nullable=False)
    name       = db.Column(db.String(100), nullable=False)
    section    = db.Column(db.String(10), nullable=False)
    email      = db.Column(db.String(120), default='')
    phone      = db.Column(db.String(15), default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    embeddings  = db.relationship('FaceEmbedding', backref='student', cascade='all, delete')
    enrollments = db.relationship('Enrollment', backref='student', cascade='all, delete')
    attendances = db.relationship('Attendance', backref='student', cascade='all, delete')

    def to_dict(self):
        return {
            'id': self.id, 'roll_no': self.roll_no, 'name': self.name,
            'section': self.section, 'email': self.email, 'phone': self.phone,
            'enrolled': len(self.enrollments) > 0,
            'has_face': len(self.embeddings) > 0
        }


class Subject(db.Model):
    __tablename__ = 'subjects'
    id      = db.Column(db.Integer, primary_key=True)
    code    = db.Column(db.String(20), unique=True, nullable=False)
    name    = db.Column(db.String(100), nullable=False)
    faculty = db.Column(db.String(100), default='')
    section = db.Column(db.String(10), default='')

    enrollments = db.relationship('Enrollment', backref='subject', cascade='all, delete')
    attendances = db.relationship('Attendance', backref='subject', cascade='all, delete')

    def to_dict(self):
        return {
            'id': self.id, 'code': self.code, 'name': self.name,
            'faculty': self.faculty, 'section': self.section,
            'enrolled_count': len(self.enrollments)
        }


class Enrollment(db.Model):
    __tablename__ = 'enrollments'
    id         = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('student_id', 'subject_id'),)


class FaceEmbedding(db.Model):
    __tablename__ = 'face_embeddings'
    id             = db.Column(db.Integer, primary_key=True)
    student_id     = db.Column(db.Integer, db.ForeignKey('students.id'), unique=True)
    embedding_path = db.Column(db.String(255))   # path to .npy file
    frame_count    = db.Column(db.Integer, default=0)
    updated_at     = db.Column(db.DateTime, default=datetime.utcnow)


class Attendance(db.Model):
    __tablename__ = 'attendance'
    id         = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subjects.id'), nullable=False)
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)
    date       = db.Column(db.Date, default=date.today)
    confidence = db.Column(db.Float, default=0.0)
    method     = db.Column(db.String(50), default='ArcFace')
    spoof_score = db.Column(db.Float, default=0.0)

    __table_args__ = (db.UniqueConstraint('student_id', 'subject_id', 'date'),)
