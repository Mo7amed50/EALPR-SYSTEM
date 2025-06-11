## import_visitors.py

from mongoengine import connect, disconnect
from models import Visitor
from datetime import datetime, timedelta
import csv
import random
from config import MONGODB_URI, MONGODB_DB_NAME


def generate_egyptian_name():
    first_names = ['محمد', 'علي', 'أحمد', 'فاطمة', 'سارة', 'خالد', 'عمر', 'نور', 'ليلى', 'ريم',
                   'محمود', 'حسن', 'عبدالله', 'مصطفى', 'يوسف', 'عبدالرحمن', 'عبدالعزيز', 'عبدالوهاب']
    last_names = ['عبدالله', 'محمد', 'علي', 'حسن', 'محمود', 'أحمد', 'عبدالرحمن', 'عبدالعزيز',
                  'عبدالوهاب', 'عبدالسلام', 'عبدالرزاق', 'عبدالفتاح', 'عبدالمنعم']
    return f"{random.choice(first_names)} {random.choice(last_names)}"


# أقسام نموذجية
responsible_departments = ['Security', 'Administration', 'IT Department', 'Maintenance', 'Human Resources']
general_departments = ['General Affairs', 'Operations', 'Support Services', 'Management']


def import_visitors():
    print("🔌 Connecting to MongoDB...")
    connect(MONGODB_DB_NAME, host=MONGODB_URI)

    try:
        print("📂 Reading CSV data...")
        plates = []
        with open('detailed_results.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Detected_Text']:
                    plates.append({
                        'plate_number': row['Detected_Text'],
                        'confidence': float(row['Confidence_Score']),
                        'image_path': row['Image_Path']
                    })

        print(f"🔢 Found {len(plates)} license plates in CSV")

        processed_plates = set()
        visitors_created = 0

        print("📥 Importing visitor records...\n")

        # تحديد تاريخ اليوم العالمي (UTC) لجميع الزوار في هذه الجولة
        today_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        for plate_data in plates:
            if plate_data['plate_number'] in processed_plates:
                continue

            processed_plates.add(plate_data['plate_number'])

            visitor = Visitor.objects(license_plate=plate_data['plate_number']).first()

            if not visitor:
                name = generate_egyptian_name()

                # إضافة وقت عشوائي ضمن اليوم نفسه (UTC)
                random_time = timedelta(
                    hours=random.randint(8, 18),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                entry_datetime_utc = today_utc + random_time

                exit_time = entry_datetime_utc + timedelta(hours=random.randint(1, 4)) if random.random() > 0.3 else None
                authorized = random.random() > 0.2  # 80% مسموح لهم

                visitor = Visitor(
                    name=name,
                    license_plate=plate_data['plate_number'],
                    entry_datetime_utc=entry_datetime_utc,
                    exit_time=exit_time,
                    authorized=authorized,
                    status='authorized' if authorized else 'unauthorized',
                    responsible_department=random.choice(responsible_departments),
                    general_department=random.choice(general_departments)
                )
                visitor.save()
                visitors_created += 1

                print(f"✅ Created: {name} | ID: {visitor.visitor_id} | Code: {visitor.visitor_code}")
                print(f"   📅 Date: {visitor.entry_date}, Time: {visitor.entry_time}")

                if visitors_created % 100 == 0:
                    print(f"🔄 Created {visitors_created} visitors so far...")

        print("\n🎉 Import completed successfully!")
        print("📊 Summary:")
        print(f"- Total unique plates found: {len(processed_plates)}")
        print(f"- New visitors created: {visitors_created}")
        print(f"- Existing visitors skipped: {len(processed_plates) - visitors_created}")

    except Exception as e:
        print(f"\n❌ Error during import: {str(e)}")
        raise
    finally:
        disconnect()
        print("\n🔌 Disconnected from MongoDB")


if __name__ == "__main__":
    import_visitors()