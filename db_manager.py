from mongoengine import connect, disconnect
from models import User, Visitor, UserActivity, SystemSettings, DetectionResult
from datetime import datetime, timedelta
from config import MONGODB_URI, MONGODB_DB_NAME

class DatabaseManager:
    def __init__(self):
        connect(MONGODB_DB_NAME, host=MONGODB_URI)
    
    def __del__(self):
        disconnect()
    
    def get_system_stats(self):
        """Get system statistics"""
        return {
            'total_users': User.objects.count(),
            'total_visitors': Visitor.objects.count(),
            'authorized_visitors': Visitor.objects(authorized=True).count(),
            'total_detections': DetectionResult.objects.count(),
            'recent_detections': DetectionResult.objects(timestamp__gte=datetime.utcnow() - timedelta(days=7)).count(),
            'total_activities': UserActivity.objects.count()
        }
    
    def cleanup_old_data(self, days=30):
        """Clean up old detection results and activities"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old detection results
        old_detections = DetectionResult.objects(timestamp__lt=cutoff_date)
        detection_count = old_detections.count()
        old_detections.delete()
        
        # Delete old activities
        old_activities = UserActivity.objects(timestamp__lt=cutoff_date)
        activity_count = old_activities.count()
        old_activities.delete()
        
        return {
            'deleted_detections': detection_count,
            'deleted_activities': activity_count
        }
    
    def backup_database(self):
        """Create a backup of important data"""
        backup = {
            'users': list(User.objects.all().values_list()),
            'visitors': list(Visitor.objects.all().values_list()),
            'settings': list(SystemSettings.objects.all().values_list()),
            'timestamp': datetime.utcnow()
        }
        return backup
    
    def reset_database(self):
        """Reset the database (use with caution!)"""
        UserActivity.objects.delete()
        DetectionResult.objects.delete()
        Visitor.objects.delete()
        User.objects.delete()
        SystemSettings.objects.delete()
    
    def create_test_data(self):
        """Create some test data for development"""
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

if __name__ == "__main__":
    manager = DatabaseManager()
    
    # Example usage:
    print("System Statistics:")
    print(manager.get_system_stats())
    
    # Uncomment the following lines to perform specific actions:
    # print(manager.cleanup_old_data(days=30))  # Clean up data older than 30 days
    print(manager.create_test_data())  # Create test data
    print(manager.backup_database())  # Create a backup
    # manager.reset_database()  # Reset the database (use with caution!) 