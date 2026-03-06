# cleanup_db.py

from mongoengine import connect, disconnect, get_connection
from models import User, Visitor, UserActivity, SystemSettings, DetectionResult
from config import MONGODB_URI, MONGODB_DB_NAME


def cleanup_database():
    print("🔌 Connecting to MongoDB...")
    try:
        disconnect(alias="default")
    except Exception:
        pass
    connect(
        MONGODB_DB_NAME,
        host=MONGODB_URI,
        alias="default",
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )

    try:
        client = get_connection()
        db = client[MONGODB_DB_NAME]

        # Drop visitor collection and remove problematic indexes
        print("🧹 Dropping collections...")
        db.visitor.drop()  # This also removes all indexes
        db.detection_result.drop()
        db.user_activity.drop()
        db.system_settings.drop()
        print("✓ All problem-causing collections dropped")

        # Delete users except admin
        print("Deleting users...")
        User.objects(username__ne='admin').delete()
        print("✓ Deleted all users except admin")

        print("\n✅ Database cleanup completed successfully!")
        print("\nCollections cleaned:")
        print("1. Visitors - Collection dropped (including bad indexes)")
        print("2. Detection Results - Collection dropped")
        print("3. User Activities - Collection dropped")
        print("4. System Settings - Collection dropped")
        print("5. Users - All users deleted except admin")

    except Exception as e:
        print(f"\n❌ Error during database cleanup: {str(e)}")
        raise
    finally:
        disconnect()


if __name__ == '__main__':
    cleanup_database()