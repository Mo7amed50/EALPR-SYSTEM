import threading
import atexit
from contextlib import contextmanager
import importlib
from mongoengine import connect, disconnect, get_connection
from config import MONGODB_URI, MONGODB_DB_NAME
from datetime import datetime, timedelta

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()
    _connected = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def ensure_connection(self):
        with self._lock:
            if not self._connected:
                try:
                    disconnect('default')
                except:
                    pass
                connect(
                    MONGODB_DB_NAME,
                    host=MONGODB_URI,
                    alias='default',
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=20000,
                    socketTimeoutMS=20000
                )
                self._connected = True

    @contextmanager
    def connection(self):
        self.ensure_connection()
        try:
            yield
        finally:
            pass

    def get_models(self):
        self.ensure_connection()
        models = importlib.import_module('models')
        return {
            'User': models.User,
            'Visitor': models.Visitor,
            'UserActivity': models.UserActivity,
            'SystemSettings': models.SystemSettings,
            'DetectionResult': models.DetectionResult
        }

    # Existing methods refactored to use self.connection()
    def get_system_stats(self):
        """Get system statistics"""
        models = self.get_models()
        User = models['User']
        Visitor = models['Visitor']
        DetectionResult = models['DetectionResult']
        UserActivity = models['UserActivity']
        with self.connection():
            return {
                'total_users': User.objects.count(),
                'total_visitors': Visitor.objects.count(),
                'authorized_visitors': Visitor.objects(authorized=True).count(),
                'total_detections': DetectionResult.objects.count(),
                'recent_detections': DetectionResult.objects(timestamp__gte=datetime.utcnow() - timedelta(days=7)).count(),
                'total_activities': UserActivity.objects.count()
            }

    # ... rest of methods similarly wrapped

    def cleanup_old_data(self, days=30):
        """Clean up old detection results and activities"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        with self.connection():
            old_detections = DetectionResult.objects(timestamp__lt=cutoff_date)
            detection_count = old_detections.count()
            old_detections.delete()
            
            old_activities = UserActivity.objects(timestamp__lt=cutoff_date)
            activity_count = old_activities.count()
            old_activities.delete()
            
            return {
                'deleted_detections': detection_count,
                'deleted_activities': activity_count
            }

    def backup_database(self):
        """Create a backup of important data"""
        with self.connection():
            backup = {
                'users': [u.to_mongo().__dict__ for u in User.objects.all()],
                'visitors': [v.to_mongo().__dict__ for v in Visitor.objects.all()],
                'settings': [s.to_mongo().__dict__ for s in SystemSettings.objects.all()],
                'timestamp': datetime.utcnow()
            }
            return backup

    def reset_database(self):
        """Reset the database (use with caution!)"""
        with self.connection():
            UserActivity.objects.delete()
            DetectionResult.objects.delete()
            Visitor.objects.delete()
            User.objects.delete()
            SystemSettings.objects.delete()

    def create_test_data(self):
        """Create some test data for development"""
        with self.connection():
            # Create test users
            test_users = [
                {'username': 'test1', 'password': 'test123', 'is_admin': False},
                {'username': 'test2', 'password': 'test123', 'is_admin': False}
            ]
            
            for user_data in test_users:
                if not User.objects(username=user_data['username']).first():
                    user = User(
                        username=user_data['username'],
                        is_admin=user_data['is_admin']
                    )
                    user.set_password(user_data['password'])
                    user.save()
            
            # Create test visitors
            test_visitors = [
                {
                    'name': 'Test Visitor 1',
                    'id_number': 'ID001',
                    'license_plate': 'ABC123',
                    'authorized': True
                },
                {
                    'name': 'Test Visitor 2',
                    'id_number': 'ID002',
                    'license_plate': 'XYZ789',
                    'authorized': False
                }
            ]
            
            for visitor_data in test_visitors:
                if not Visitor.objects(id_number=visitor_data['id_number']).first():
                    Visitor(**visitor_data).save()
            
            return "Test data created successfully!"

# Global singleton
db_manager = DatabaseManager()

if __name__ == "__main__":
    print("System Statistics:", db_manager.get_system_stats())
    print(db_manager.create_test_data())

atexit.register(lambda: disconnect('default'))

