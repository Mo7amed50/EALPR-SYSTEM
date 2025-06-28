# models.py

from mongoengine import (
    connect, Document, StringField,
    DateTimeField, BooleanField, SequenceField,
    ReferenceField, CASCADE, IntField,
    FloatField, BinaryField 
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from datetime import datetime
from config import MONGODB_URI, MONGODB_DB_NAME
import pytz 


EGYPT_TZ = pytz.timezone('Africa/Cairo')


# الاتصال بقاعدة البيانات
connect(MONGODB_DB_NAME, host=MONGODB_URI, alias='default')


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
        return str(self.id)

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
        if not self.visitor_code and self.visitor_id:
            self.visitor_code = f"V{str(self.visitor_id).zfill(3)}"

        # تحويل entry_datetime_utc إلى منطقة زمنية محلية
        if self.entry_datetime_utc:
            utc_time = pytz.utc.localize(self.entry_datetime_utc)
            local_time = utc_time.astimezone(EGYPT_TZ)

            self.entry_date = local_time.strftime('%Y-%m-%d')
            self.entry_time = local_time.strftime('%H:%M:%S')

        super(Visitor, self).save(*args, **kwargs)
    

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
        setting = SystemSettings.objects(key=key).first()
        return setting.value if setting else default

    @staticmethod
    def set_setting(key, value, description=None, user_id=None):
        setting = SystemSettings.objects(key=key).first()
        if setting:
            setting.value = value
            if user_id:
                setting.updated_by = User.objects(id=user_id).first()
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
                setting.updated_by = User.objects(id=user_id).first()
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

    meta = {
        'indexes': ['plate_number', 'timestamp', 'status']
    }