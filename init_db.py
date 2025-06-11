# init_db.py

from mongoengine import connect, disconnect
from models import User, Visitor, UserActivity, SystemSettings, DetectionResult
from datetime import datetime
from config import MONGODB_URI, MONGODB_DB_NAME


def init_database():
    print("🔌 Connecting to MongoDB...")
    connect(MONGODB_DB_NAME, host=MONGODB_URI)

    try:
        # Create indexes for User collection
        print("Creating User collection and indexes...")
        User.objects._collection.create_index("username", unique=True)
        User.objects._collection.create_index("is_admin")
        User.objects._collection.create_index("last_login")
        print("✓ User collection created")

        # Create indexes for Visitor collection
        print("Creating Visitor collection and indexes...")
        Visitor.objects._collection.create_index("license_plate", unique=True)
        Visitor.objects._collection.create_index("entry_datetime_utc")
        Visitor.objects._collection.create_index("authorized")
        Visitor.objects._collection.create_index("status")
        Visitor.objects._collection.create_index("visitor_code", unique=True)
        print("✓ Visitor collection created")

        # Create indexes for UserActivity collection
        print("Creating UserActivity collection and indexes...")
        UserActivity.objects._collection.create_index("user")
        UserActivity.objects._collection.create_index("timestamp")
        UserActivity.objects._collection.create_index("action")
        print("✓ UserActivity collection created")

        # Create indexes for SystemSettings collection
        print("Creating SystemSettings collection and indexes...")
        SystemSettings.objects._collection.create_index("key", unique=True)
        SystemSettings.objects._collection.create_index("updated_at")
        print("✓ SystemSettings collection created")

        # Create indexes for DetectionResult collection
        print("Creating DetectionResult collection and indexes...")
        DetectionResult.objects._collection.create_index("timestamp")
        DetectionResult.objects._collection.create_index("plate_number")
        DetectionResult.objects._collection.create_index("status")
        DetectionResult.objects._collection.create_index("processed_by")
        print("✓ DetectionResult collection created")

        # Create default admin user if not exists
        print("Creating default admin user...")
        if not User.objects(username='admin').first():
            admin = User(
                username='admin',
                is_admin=True,
                created_at=datetime.utcnow()
            )
            admin.set_password('admin123')
            admin.save()
            print("✓ Default admin user created")

        # Create default system settings
        print("Creating default system settings...")
        default_settings = [
            {
                'key': 'confidence_threshold',
                'value': '0.5',
                'description': 'Minimum confidence threshold for plate detection'
            },
            {
                'key': 'max_image_size',
                'value': '5242880',  # 5MB in bytes
                'description': 'Maximum allowed image size in bytes'
            },
            {
                'key': 'allowed_extensions',
                'value': 'jpg,jpeg,png',
                'description': 'Allowed image file extensions'
            },
            {
                'key': 'system_name',
                'value': 'EALPR System',
                'description': 'Name of the system'
            },
            {
                'key': 'maintenance_mode',
                'value': 'false',
                'description': 'System maintenance mode'
            }
        ]

        for setting in default_settings:
            if not SystemSettings.objects(key=setting['key']).first():
                SystemSettings(
                    key=setting['key'],
                    value=setting['value'],
                    description=setting['description'],
                    updated_at=datetime.utcnow()
                ).save()
        print("✓ Default system settings created")

        print("\n🎉 Database initialization completed successfully!")
        print("\nCollections created:")
        print("1. Users - For system users and administrators")
        print("2. Visitors - For authorized and unauthorized visitors")
        print("3. UserActivity - For tracking user actions")
        print("4. SystemSettings - For system configuration")
        print("5. DetectionResult - For storing license plate detection results")

    except Exception as e:
        print(f"\n❌ Error during database initialization: {str(e)}")
        raise
    finally:
        disconnect()


if __name__ == '__main__':
    init_database()