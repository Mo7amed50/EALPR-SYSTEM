# models.py

from mongoengine import (
    Document, StringField,
    DateTimeField, BooleanField, SequenceField,
    ReferenceField, CASCADE, IntField,
    FloatField, BinaryField 
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime
import pytz 


EGYPT_TZ = pytz.timezone('Africa/Cairo')

class User(Document, UserMixin):
    username = StringField(required=True, unique=True)
    password = StringField(required=True)
    is_admin = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField()
    is_active = BooleanField(default=True)
    failed_login_attempts = IntField(default=0)
    last_failed_login = DateTimeField()

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def get_id(self):
        return str(getattr(self, "id", ""))

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False


class Visitor(Document):
    visitor_id = SequenceField(unique=True)  # الرقم التسلسلي مثل: 1, 2, 3...
    visitor_code = StringField(unique=True)  # الكود الجديد مثل: V001, V002...

    name = StringField(required=True)
    id_number = StringField()  # optional national ID, used by some scripts
    license_plate = StringField(required=True, unique=True)

    entry_datetime_utc = DateTimeField(default=datetime.utcnow)  # التاريخ بالتوقيت العالمي
    entry_date = StringField()  # التاريخ فقط - محول للتوقيت المحلي
    entry_time = StringField()  # الوقت فقط - محول للتوقيت المحلي

    exit_time = DateTimeField()
    authorized = BooleanField(default=False)
    status = StringField(default='pending')  # pending, authorized, unauthorized
    responsible_department = StringField()
    general_department = StringField()

    def save(self, *args, **kwargs):
        # Convert entry_datetime_utc to local Cairo time
        if self.entry_datetime_utc:
            utc_time = self.entry_datetime_utc
            if utc_time.tzinfo is None:
                utc_time = pytz.utc.localize(utc_time)
            local_time = utc_time.astimezone(EGYPT_TZ)

            self.entry_date = local_time.strftime('%Y-%m-%d')
            self.entry_time = local_time.strftime('%H:%M:%S')

        # If visitor_code is not assigned, and visitor_id is already assigned
        if not self.visitor_code and self.visitor_id:
            self.visitor_code = f"V{str(self.visitor_id).zfill(3)}"

        ret = super(Visitor, self).save(*args, **kwargs)

        # For newly created documents where visitor_id was assigned by SequenceField during save
        if not self.visitor_code and self.visitor_id:
            self.visitor_code = f"V{str(self.visitor_id).zfill(3)}"
            super(Visitor, self).save(*args, **kwargs)

        return ret
    

class UserActivity(Document):
    user = ReferenceField(User, required=True, reverse_delete_rule=CASCADE)
    action = StringField(required=True)
    details = StringField()
    ip_address = StringField()
    timestamp = DateTimeField(default=datetime.utcnow)

    meta = {
        'indexes': ['user', 'timestamp', 'action'],
        'ordering': ['-timestamp']
    }


class SystemSettings(Document):
    key = StringField(required=True, unique=True)
    value = StringField()
    description = StringField()
    updated_at = DateTimeField(default=datetime.utcnow)
    updated_by = ReferenceField(User)

    @staticmethod
    def get_setting(key, default=None):
        setting = getattr(SystemSettings, "objects")(key=key).first()
        return setting.value if setting else default

    @staticmethod
    def set_setting(key, value, description=None, user_id=None):
        setting = getattr(SystemSettings, "objects")(key=key).first()
        if setting:
            setting.value = value
            if user_id:
                setting.updated_by = getattr(User, "objects")(id=user_id).first()
            if description:
                setting.description = description
            setting.updated_at = datetime.utcnow()
        else:
            setting = SystemSettings(
                key=key,
                value=value,
                description=description
            )
            if user_id:
                setting.updated_by = getattr(User, "objects")(id=user_id).first()
        setting.save()
        return setting


class DetectionResult(Document):
    timestamp = DateTimeField(default=datetime.utcnow)
    plate_number = StringField()
    confidence = FloatField()
    status = StringField()  # authorized/unauthorized
    visitor_name = StringField()
    processed_by = ReferenceField(User)
    original_image = BinaryField()
    processed_image = BinaryField()
    original_image_path = StringField()
    processed_image_path = StringField()
    ocr_confidence = FloatField()


    meta = {
        'indexes': ['plate_number', 'timestamp', 'status']
    }


# ── Phase 2 / Block C2 — asynchronous inference jobs ─────────────────────────
#
# New, isolated collection. DetectionResult and Visitor are NOT modified: this
# document only *references* an existing detection once inference succeeds.
# No migration is required and no existing record is touched.
#
# Field reference:
#   job_id             Stable public identifier returned to the client. UUID4
#                      hex, unique and indexed. This is the only job identifier
#                      ever exposed over HTTP (never the Mongo _id).
#   user               The authenticated user who requested the job. Enforces
#                      job ownership; the worker writes the same document into
#                      DetectionResult.processed_by.
#   status             One of: queued | processing | completed | failed.
#   image_path         Absolute path to the staged upload on the server. NEVER
#                      serialised into any API response.
#   original_filename  Client-supplied filename, sanitised. For display only.
#   created_at         When the job was accepted by POST /api/inference/jobs.
#   started_at         When the worker actually began inference.
#   completed_at       When the job reached a terminal state (either outcome).
#   detection          The DetectionResult produced on success. Null until then.
#   plate_text         Recognised plate, for a quick status read without
#                      loading the full detection document.
#   error_code         Stable machine-readable failure category for the client.
#   error_message      Sanitised, human-readable failure text safe to expose.
#                      Never contains a traceback, path, or driver message.
#   attempts           Number of worker attempts, for diagnostics.
#
# The full exception is written to the server log, never to this document, so a
# failed job cannot leak internals through the status endpoint.
class InferenceJob(Document):
    STATUS_QUEUED = "queued"
    STATUS_PROCESSING = "processing"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    job_id = StringField(required=True, unique=True)
    user = ReferenceField(User)
    status = StringField(
        required=True,
        default=STATUS_QUEUED,
        choices=[STATUS_QUEUED, STATUS_PROCESSING, STATUS_COMPLETED, STATUS_FAILED],
    )
    image_path = StringField()
    original_filename = StringField()
    created_at = DateTimeField(default=datetime.utcnow, required=True)
    started_at = DateTimeField()
    completed_at = DateTimeField()
    detection = ReferenceField(DetectionResult)
    plate_text = StringField()
    error_code = StringField()
    error_message = StringField()
    attempts = IntField(default=0)

    meta = {
        'indexes': ['job_id', 'user', 'status', 'created_at'],
        'ordering': ['-created_at'],
    }