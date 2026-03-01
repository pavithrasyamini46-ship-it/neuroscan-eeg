"""
app.py - Main Flask Application
EEG Neurological Disorder Diagnosis System
Secure, Role-Based, AES-Encrypted, ML-Powered
"""

import os
import json
import time
import uuid
import logging
import numpy as np
import pandas as pd
import joblib

from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, session, jsonify, send_file, abort)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                          logout_user, login_required, current_user)
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

from encryption import encrypt_file, decrypt_file, encrypt_report, decrypt_report
from features import extract_all_features

# ─────────────────────────────────────────────────────────────
# APP CONFIGURATION
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY']            = os.environ.get('SECRET_KEY', 'eeg-secure-2024-xk92')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER']         = 'static/uploads'
app.config['REPORTS_FOLDER']        = 'static/reports'
app.config['ENCRYPTED_FOLDER']      = 'encrypted_data'
app.config['MAX_CONTENT_LENGTH']    = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'csv'}

db           = SQLAlchemy(app)
bcrypt       = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure folders exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'],
               app.config['ENCRYPTED_FOLDER'], 'logs', 'models']:
    os.makedirs(folder, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename='logs/app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ─────────────────────────────────────────────────────────────
# DATABASE MODELS
# ─────────────────────────────────────────────────────────────

class User(db.Model, UserMixin):
    """User model with role-based access."""
    id           = db.Column(db.Integer, primary_key=True)
    username     = db.Column(db.String(80),  unique=True, nullable=False)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    role         = db.Column(db.String(20),  nullable=False, default='doctor')  # 'admin' or 'doctor'
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    reports      = db.relationship('Report', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username} [{self.role}]>'


class Report(db.Model):
    """Stores encrypted prediction reports."""
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    filename     = db.Column(db.String(200), nullable=False)
    enc_file     = db.Column(db.String(200))              # Path to encrypted EEG file
    prediction   = db.Column(db.String(100))              # Predicted class
    model_used   = db.Column(db.String(50))               # Which model was used
    confidence   = db.Column(db.Float)                    # Confidence %
    enc_report   = db.Column(db.LargeBinary)              # Encrypted full report blob
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)


class SystemLog(db.Model):
    """Audit log for security and system events."""
    id         = db.Column(db.Integer, primary_key=True)
    user       = db.Column(db.String(80))
    action     = db.Column(db.String(200))
    ip         = db.Column(db.String(50))
    timestamp  = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_action(action: str):
    """Write an audit log entry to DB and file."""
    entry = SystemLog(
        user=current_user.username if not current_user.is_anonymous else 'anonymous',
        action=action,
        ip=request.remote_addr
    )
    db.session.add(entry)
    db.session.commit()
    logging.info(f"[{entry.user}] {action}")

def role_required(*roles):
    """Decorator to restrict route access by role."""
    def decorator(f):
        @wraps(f)
        @login_required
        def decorated(*args, **kwargs):
            if current_user.role not in roles:
                flash('Access denied: insufficient permissions.', 'danger')
                abort(403)
            return f(*args, **kwargs)
        return decorated
    return decorator

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────

def load_ml_model(model_name: str = None):
    """Load a trained ML model by name. Defaults to best model."""
    if model_name is None:
        best_file = 'models/best_model.txt'
        if os.path.exists(best_file):
            with open(best_file) as f:
                model_name = f.read().strip()
        else:
            model_name = 'Random Forest'

    map_files = {
        'Random Forest': 'models/rf_model.pkl',
        'SVM':           'models/svm_model.pkl',
        'XGBoost':       'models/xgb_model.pkl',
    }

    path = map_files.get(model_name, 'models/rf_model.pkl')
    if not os.path.exists(path):
        # Fall back to any available model
        for p in map_files.values():
            if os.path.exists(p):
                return joblib.load(p), os.path.basename(p).replace('_model.pkl','').title()
        return None, None

    return joblib.load(path), model_name

def load_scaler():
    path = 'models/scaler.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_label_encoder():
    path = 'models/label_encoder.pkl'
    if os.path.exists(path):
        return joblib.load(path)
    return None

def predict_from_csv(csv_path: str, model_name: str = None):
    """
    Load EEG CSV, extract features, run prediction.
    Returns: prediction label, confidence, feature dict.
    """
    model, used_name = load_ml_model(model_name)
    scaler = load_scaler()
    le     = load_label_encoder()

    if model is None:
        raise RuntimeError("No trained model found. Please run train_model.py first.")

    df = pd.read_csv(csv_path)

    # Remove label column if present
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'y' in df.columns:
        df = df.drop(columns=['y'])

    # Use first row for single-sample prediction
    signal = df.iloc[0].values.astype(np.float32)

    # Extract features
    features = extract_all_features(signal).reshape(1, -1)

    if scaler is not None:
        features = scaler.transform(features)

    # Predict
    pred_idx = model.predict(features)[0]
    proba    = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        confidence = float(proba[pred_idx]) * 100
    else:
        confidence = 0.0

    if le is not None:
        pred_label = str(le.inverse_transform([pred_idx])[0])
    else:
        pred_label = str(pred_idx)

    feat_dict = {
        'mean':      float(np.mean(signal)),
        'variance':  float(np.var(signal)),
        'std':       float(np.std(signal)),
        'energy':    float(np.sum(signal**2)),
    }

    return pred_label, confidence, feat_dict, used_name

# ─────────────────────────────────────────────────────────────
# ROUTES: PUBLIC
# ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

# ─────────────────────────────────────────────────────────────
# ROUTES: AUTH
# ─────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        role     = request.form.get('role', 'doctor')

        # Basic validation
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')

        # Only allow admin role if no admin exists yet OR via a secret token
        if role == 'admin':
            admin_token = request.form.get('admin_token', '')
            if admin_token != app.config['SECRET_KEY'][:8]:
                role = 'doctor'  # Downgrade to doctor if token wrong

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_pw, role=role)
        db.session.add(user)
        db.session.commit()

        logging.info(f"New user registered: {username} [{role}]")
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user     = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user, remember=True)
            logging.info(f"User logged in: {username} [{user.role}] from {request.remote_addr}")
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            logging.warning(f"Failed login attempt for: {username} from {request.remote_addr}")

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logging.info(f"User logged out: {current_user.username}")
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Route to appropriate dashboard based on role."""
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    return redirect(url_for('doctor_dashboard'))

# ─────────────────────────────────────────────────────────────
# ROUTES: ADMIN
# ─────────────────────────────────────────────────────────────

@app.route('/admin')
@role_required('admin')
def admin_dashboard():
    users   = User.query.all()
    logs    = SystemLog.query.order_by(SystemLog.timestamp.desc()).limit(50).all()
    reports = Report.query.order_by(Report.created_at.desc()).all()

    # Load model metrics from training
    metrics = []
    if os.path.exists('models/metrics.json'):
        with open('models/metrics.json') as f:
            metrics = json.load(f)

    total_users   = len(users)
    total_reports = len(reports)
    log_action("Viewed admin dashboard")
    return render_template('admin_dashboard.html',
                           users=users, logs=logs, metrics=metrics,
                           total_users=total_users, total_reports=total_reports,
                           reports=reports)


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@role_required('admin')
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot delete your own account.', 'warning')
        return redirect(url_for('admin_dashboard'))
    db.session.delete(user)
    db.session.commit()
    log_action(f"Deleted user: {user.username}")
    flash(f'User {user.username} deleted.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/metrics_json')
@role_required('admin')
def metrics_json():
    """API endpoint for Chart.js model performance charts."""
    if os.path.exists('models/metrics.json'):
        with open('models/metrics.json') as f:
            return jsonify(json.load(f))
    return jsonify([])

# ─────────────────────────────────────────────────────────────
# ROUTES: DOCTOR
# ─────────────────────────────────────────────────────────────

@app.route('/doctor')
@role_required('doctor', 'admin')
def doctor_dashboard():
    reports = Report.query.filter_by(user_id=current_user.id)\
                          .order_by(Report.created_at.desc()).all()
    log_action("Viewed doctor dashboard")
    return render_template('doctor_dashboard.html', reports=reports)


@app.route('/upload', methods=['GET', 'POST'])
@role_required('doctor', 'admin')
def upload():
    if request.method == 'POST':
        # Validate file
        if 'eeg_file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)

        f = request.files['eeg_file']
        if f.filename == '' or not allowed_file(f.filename):
            flash('Invalid file. Please upload a CSV file.', 'danger')
            return redirect(request.url)

        patient_name = request.form.get('patient_name', 'Unknown').strip()
        model_choice = request.form.get('model', None)

        # Save uploaded file
        uid      = str(uuid.uuid4())[:8]
        filename = secure_filename(f.filename)
        raw_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{uid}_{filename}')
        f.save(raw_path)

        try:
            # Encrypt the uploaded EEG file
            enc_path = encrypt_file(raw_path)

            # Run prediction (on unencrypted temp file)
            pred_label, confidence, feat_dict, used_model = predict_from_csv(
                raw_path, model_name=model_choice)

            # Build full report dict
            report_data = {
                'patient':     patient_name,
                'prediction':  pred_label,
                'confidence':  confidence,
                'model':       used_model,
                'features':    feat_dict,
                'timestamp':   datetime.utcnow().isoformat(),
                'doctor':      current_user.username,
            }

            # Encrypt the report blob
            enc_report_bytes = encrypt_report(report_data)

            # Save to DB
            report = Report(
                user_id      = current_user.id,
                patient_name = patient_name,
                filename     = filename,
                enc_file     = enc_path,
                prediction   = pred_label,
                model_used   = used_model,
                confidence   = confidence,
                enc_report   = enc_report_bytes
            )
            db.session.add(report)
            db.session.commit()

            # Remove raw (unencrypted) file from uploads
            if os.path.exists(raw_path):
                os.remove(raw_path)

            log_action(f"Uploaded EEG for patient '{patient_name}' → Prediction: {pred_label}")
            flash('EEG file processed successfully!', 'success')
            return redirect(url_for('result', report_id=report.id))

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            flash(f'Error during prediction: {str(e)}', 'danger')
            if os.path.exists(raw_path):
                os.remove(raw_path)
            return redirect(url_for('upload'))

    return render_template('upload.html')


@app.route('/result/<int:report_id>')
@role_required('doctor', 'admin')
def result(report_id):
    report = Report.query.get_or_404(report_id)

    # Doctors can only view their own reports
    if current_user.role == 'doctor' and report.user_id != current_user.id:
        abort(403)

    # Decrypt the full report
    try:
        report_data = decrypt_report(report.enc_report)
    except Exception:
        report_data = {'prediction': report.prediction, 'confidence': report.confidence}

    log_action(f"Viewed result for report ID {report_id}")
    return render_template('result.html', report=report, report_data=report_data)


@app.route('/reports')
@role_required('doctor', 'admin')
def reports():
    if current_user.role == 'admin':
        all_reports = Report.query.order_by(Report.created_at.desc()).all()
    else:
        all_reports = Report.query.filter_by(user_id=current_user.id)\
                                   .order_by(Report.created_at.desc()).all()
    log_action("Viewed reports page")
    return render_template('reports.html', reports=all_reports)


@app.route('/report/download/<int:report_id>')
@role_required('doctor', 'admin')
def download_report(report_id):
    from docx import Document as DocxDocument
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    report = Report.query.get_or_404(report_id)

    if current_user.role == 'doctor' and report.user_id != current_user.id:
        abort(403)

    try:
        report_data = decrypt_report(report.enc_report)
    except Exception:
        report_data = {}

    # ── Build a professional Word document ──
    doc = DocxDocument()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin   = Inches(1.2)
        section.right_margin  = Inches(1.2)

    # ── HEADER ──
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = header_para.add_run("NEUROSCAN EEG")
    run.bold      = True
    run.font.size = Pt(20)
    run.font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("Multidimensional Neurological Disorder Diagnosis System")
    r.font.size  = Pt(11)
    r.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()  # spacer

    # ── TITLE ──
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t = title.add_run("EEG DIAGNOSIS REPORT")
    t.bold = True
    t.font.size = Pt(16)
    t.font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    doc.add_paragraph()

    # ── HORIZONTAL LINE ──
    p = doc.add_paragraph()
    p.add_run("─" * 65)

    # ── HELPER: add label-value row ──
    def add_row(label, value, bold_value=False, color=None):
        p = doc.add_paragraph()
        r1 = p.add_run(f"{label}: ")
        r1.bold = True
        r1.font.size = Pt(11)
        r2 = p.add_run(str(value))
        r2.bold = bold_value
        r2.font.size = Pt(11)
        if color:
            r2.font.color.rgb = color
        p.paragraph_format.space_after = Pt(4)

    # ── SECTION 1: Patient Info ──
    h1 = doc.add_heading("1. Patient Information", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    add_row("Patient Name",  report_data.get("patient", report.patient_name))
    add_row("Doctor",        f"Dr. {report_data.get('doctor', report.user.username)}")
    add_row("Report ID",     f"#{report.id}")
    add_row("Date & Time",   report.created_at.strftime("%d %B %Y, %H:%M:%S"))
    add_row("EEG File",      report.filename)

    doc.add_paragraph()

    # ── SECTION 2: Diagnosis Result ──
    h2 = doc.add_heading("2. Diagnosis Result", level=1)
    h2.runs[0].font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    prediction  = report_data.get("prediction", report.prediction)
    confidence  = report_data.get("confidence", report.confidence)
    model_used  = report_data.get("model",      report.model_used)

    pred_lower  = prediction.lower()
    is_normal   = "normal" in pred_lower
    pred_color  = RGBColor(0x28, 0xA7, 0x45) if is_normal else RGBColor(0xDC, 0x35, 0x45)

    add_row("Prediction",    prediction,  bold_value=True, color=pred_color)
    add_row("Confidence",    f"{confidence:.2f}%")
    add_row("Model Used",    model_used)
    add_row("Status",
            "NORMAL — No significant abnormality detected" if is_normal
            else "ABNORMAL — Neurological pattern detected",
            bold_value=True, color=pred_color)

    doc.add_paragraph()

    # ── SECTION 3: Clinical Note ──
    h3 = doc.add_heading("3. Clinical Note", level=1)
    h3.runs[0].font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    notes = {
        "epilep":    "Epileptic seizure pattern identified. Requires immediate neurological evaluation and possible antiepileptic therapy.",
        "alzheimer": "EEG shows cortical slowing consistent with Alzheimer's disease (increased delta/theta, reduced alpha). Recommend cognitive assessment and MRI follow-up.",
        "parkinson": "Beta-band oscillations detected, consistent with Parkinson's disease. Recommend movement disorder specialist consultation.",
        "tumor":     "Irregular EEG pattern suggesting tumor-related activity. Immediate neuroimaging (MRI/CT) recommended.",
        "normal":    "No abnormal neurological patterns detected. EEG signal is within normal range.",
    }
    note = next((v for k, v in notes.items() if k in pred_lower), "Please consult a qualified neurologist for further evaluation.")
    p = doc.add_paragraph(note)
    p.runs[0].font.size = Pt(11)
    p.runs[0].italic    = True

    doc.add_paragraph()

    # ── SECTION 4: EEG Features ──
    feats = report_data.get("features", {})
    if feats:
        h4 = doc.add_heading("4. Extracted EEG Features", level=1)
        h4.runs[0].font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

        from docx.oxml.ns import qn
        from docx.oxml   import OxmlElement

        # Feature table
        table = doc.add_table(rows=1, cols=2)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        hdr[0].text = "Feature"
        hdr[1].text = "Value"
        for cell in hdr:
            for run in cell.paragraphs[0].runs:
                run.bold = True
        for key, val in feats.items():
            row = table.add_row().cells
            row[0].text = key.replace("_", " ").title()
            row[1].text = f"{val:.6f}" if isinstance(val, float) else str(val)

        doc.add_paragraph()

    # ── SECTION 5: Security Info ──
    h5 = doc.add_heading("5. Security & Encryption", level=1)
    h5.runs[0].font.color.rgb = RGBColor(0x0D, 0x6E, 0xFD)

    add_row("Encryption",       "AES Fernet (AES-128-CBC)")
    add_row("File Storage",     "Encrypted .enc file — raw CSV deleted after processing")
    add_row("Report Storage",   "Encrypted binary blob stored in SQLite database")
    add_row("Access Control",   "Role-Based — Doctor/Admin only")
    add_row("Password Hashing", "bcrypt adaptive hashing")

    doc.add_paragraph()

    # ── FOOTER ──
    p = doc.add_paragraph()
    p.add_run("─" * 65)
    footer_p = doc.add_paragraph()
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer_p.add_run(
        "This report is generated by NeuroScan EEG — AI-Assisted Neurological Diagnosis System.\n"
        "This is NOT a substitute for clinical diagnosis by a qualified neurologist."
    )
    fr.italic    = True
    fr.font.size = Pt(9)
    fr.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

    # ── Save and send ──
    safe_name = report.patient_name.replace(" ", "_")
    tmp_path  = os.path.join(app.config['REPORTS_FOLDER'], f'report_{report_id}.docx')
    doc.save(tmp_path)

    log_action(f"Downloaded DOCX report ID {report_id}")
    return send_file(tmp_path, as_attachment=True,
                     download_name=f'EEG_Report_{safe_name}_{report_id}.docx')

# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', code=403,
                           msg='You do not have permission to access this page.'), 403

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404,
                           msg='The page you are looking for does not exist.'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500,
                           msg='An internal server error occurred.'), 500

# ─────────────────────────────────────────────────────────────
# INIT DATABASE & SEED ADMIN
# ─────────────────────────────────────────────────────────────

def seed_admin():
    """Create a default admin user if none exists."""
    if not User.query.filter_by(role='admin').first():
        hashed = bcrypt.generate_password_hash('admin123').decode('utf-8')
        admin  = User(username='admin', email='admin@eeg.local',
                      password=hashed, role='admin')
        db.session.add(admin)
        db.session.commit()
        print("[+] Default admin created → username: admin | password: admin123")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        seed_admin()
        print("[+] Database initialized.")
import os
port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)


