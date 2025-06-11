from mongoengine import connect, disconnect
from models import User, Visitor, UserActivity, SystemSettings, DetectionResult
from datetime import datetime, timedelta
import random
import string
from config import MONGODB_URI, MONGODB_DB_NAME

def generate_random_plate():
    """Generate a random Egyptian license plate number"""
    # Format: 3 digits - 2 letters - 3 digits
    numbers = ''.join(random.choices(string.digits, k=3))
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    numbers2 = ''.join(random.choices(string.digits, k=3))
    return f"{numbers}-{letters}-{numbers2}"

def generate_random_id():
    """Generate a random Egyptian ID number"""
    # Format: 14 digits
    return ''.join(random.choices(string.digits, k=14))

def generate_random_name():
    """Generate a random Egyptian name"""
    first_names = ['Ahmed', 'Mohammed', 'Ali', 'Omar', 'Mostafa', 'Ibrahim', 'Hassan', 'Mahmoud']
    last_names = ['Ali', 'Hassan', 'Ibrahim', 'Mohammed', 'Ahmed', 'Omar', 'Mostafa', 'Mahmoud']
    return f"{random.choice(first_names)} {random.choice(last_names)}"

# Sample department names (can be customized)
responsible_departments = ['Security', 'Administration', 'IT Department', 'Maintenance', 'Human Resources']
general_departments = ['General Affairs', 'Operations', 'Support Services', 'Management']

def populate_database():
    print("Connecting to MongoDB...")
    connect(MONGODB_DB_NAME, host=MONGODB_URI)
    
    try:
        # Create test users
        print("Creating test users...")
        test_users = [
            {'username': 'test1', 'password': 'test123', 'is_admin': False},
            {'username': 'test2', 'password': 'test123', 'is_admin': False}
        ]
        
        users = []
        for user_data in test_users:
            if not User.objects(username=user_data['username']).first():
                user = User(
                    username=user_data['username'],
                    is_admin=user_data['is_admin']
                )
                user.set_password(user_data['password'])
                user.save()
                users.append(user)
        print(f"✓ Created {len(users)} test users")

        # Create test visitors
        print("Creating test visitors...")
        visitors = []
        for _ in range(20):  # Create 20 test visitors
            plate = generate_random_plate()
            if not Visitor.objects(license_plate=plate).first():
                visitor = Visitor(
                    name=generate_random_name(),
                    id_number=generate_random_id(),
                    license_plate=plate,
                    entry_time=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                    exit_time=datetime.utcnow() - timedelta(hours=random.randint(1, 8)) if random.random() > 0.3 else None,
                    authorized=random.random() > 0.2,  # 80% are authorized
                    status='authorized' if random.random() > 0.2 else 'unauthorized'
                )
                # Add random department values
                visitor.responsible_department = random.choice(responsible_departments)
                visitor.general_department = random.choice(general_departments)
                visitor.save()
                visitors.append(visitor)
        print(f"✓ Created {len(visitors)} test visitors")

        # Create test detection results
        print("Creating test detection results...")
        detection_count = 0
        for visitor in visitors:
            for _ in range(random.randint(1, 5)):  # 1-5 detections per visitor
                detection = DetectionResult(
                    plate_number=visitor.license_plate,
                    confidence=random.uniform(0.7, 0.99),
                    status=visitor.status,
                    visitor_name=visitor.name,
                    timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                    processed_by=random.choice(users),
                    original_image=b"sample_image",  # Placeholder for actual image data
                    processed_image=b"sample_image"  # Placeholder for actual image data
                )
                detection.save()
                detection_count += 1
        print(f"✓ Created {detection_count} test detection results")

        # Create test user activities
        print("Creating test user activities...")
        actions = ['login', 'logout', 'detect_plate', 'add_visitor', 'update_visitor', 'view_reports']
        for _ in range(100):  # Create 100 test activities
            user = random.choice(users)
            action = random.choice(actions)
            timestamp = datetime.utcnow() - timedelta(days=random.randint(0, 30))
            
            activity = UserActivity(
                user=user,
                action=action,
                details=f"User {user.username} performed {action}",
                ip_address=f"192.168.1.{random.randint(1, 255)}",
                timestamp=timestamp
            )
            activity.save()
        print("✓ Created 100 test user activities")

        print("\nDatabase population completed successfully!")
        print("\nSummary of created data:")
        print(f"1. Users: {len(users)}")
        print(f"2. Visitors: {len(visitors)}")
        print(f"3. Detection Results: {detection_count}")
        print("4. User Activities: 100")
        
    except Exception as e:
        print(f"\nError during database population: {str(e)}")
        raise
    finally:
        disconnect()

if __name__ == '__main__':
    populate_database() 