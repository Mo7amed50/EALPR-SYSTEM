EALPR System - Egyptian Automatic License Plate Recognition
Overview
The EALPR System is a smart, AI-based application designed to automate the recognition of Egyptian vehicle license plates in real time. Developed as a graduation project by students at Helwan University's Faculty of Computers and Artificial Intelligence, this system leverages computer vision, deep learning, and database integration to streamline vehicle identification for applications such as traffic monitoring, parking management, and security systems.
The system captures live video feeds, detects license plates, recognizes Arabic characters and digits, and retrieves associated vehicle or driver information from a database. It is tailored specifically for Egyptian license plates, addressing challenges like mixed Arabic and Western numeral formats and varying plate designs.
Features

Real-Time License Plate Detection: Uses a deep learning model (YOLO) to identify and extract license plates from live video streams.
Character Recognition: Employs EasyOCR to recognize Arabic letters and digits, with preprocessing to handle the complexity of Arabic text.
Database Integration: Matches recognized plates with a MongoDB database to retrieve and display vehicle/driver details.
User-Friendly Interface: Provides live camera feed monitoring, detection history, and visitor management functionalities.
Scalable Design: Supports potential integration with smart gates, parking systems, and mobile applications.

Technologies Used

OpenCV: For video streaming, image processing, and license plate detection.
EasyOCR: For optical character recognition of Arabic and numeric characters.
YOLO (Ultralytics): For deep learning-based license plate detection and character recognition.
MongoDB: For storing and querying vehicle and visitor data.
Python: Core programming language for system implementation.
Flask/SQLAlchemy: For backend and database model management (optional relational database support planned for future versions).

Installation

Clone the Repository:
git clone https://github.com/Mo7amed50/EALPR-SYSTEM.git
cd EALPR-SYSTEM


Install Dependencies:Ensure Python 3.8+ is installed, then set up a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root and configure the following:
MODEL_PATH=working/runs/detect/yolo_car_plate/weights/best.pt
OCR_MODEL_PATH=working/runs/detect/yolov8n_car_plate/weights/best.pt
TEMP_DIR=temp
SAVE_DIR=detected_plates
TESSERACT_OCR=Path/To/Tesseract/OCR


Initialize MongoDB:Ensure MongoDB is installed and running. Run the database initialization script:
python init_database.py


Run the Application:Start the Flask server to launch the web interface:
python app.py

Access the application at http://localhost:5000.


Usage

Live Camera Feed: Start the camera to monitor real-time license plate detections.
Detection History: View past detections with timestamps, plate numbers, and visitor details.
Visitor Management: Add new visitors or manage user accounts via the interface.
Search and Reports: Filter detection history by plate number or status, and generate reports.

Project Structure

/working/runs/detect: Contains YOLO model weights for plate detection and OCR.
/temp: Temporary directory for frame processing.
/detected_plates: Stores images of detected license plates.
/app.py: Main Flask application script.
/init_database.py: Script for MongoDB initialization.
/config.py: Configuration settings for models and directories.

Future Work

Dual-Model Recognition: Implement separate English and Arabic models for improved accuracy.
Mobile App: Develop a mobile version for field use with smartphone cameras.
Relational Database: Transition to PostgreSQL or MySQL for enhanced data management.
Multimedia Logging: Add video annotation and audio transcription for detection events.

Contributors

Mohamed Mahmoud Hanafy Yassin
Mahmoud Aly Mahmoud Ahmed
Gamal El-Din Ayman Abdel-Rahman
Zeyad Ameen Hussein Masoud
Rashad Samir Rashad Eid

Supervisor: Dr. Mohammed El-Said
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
We thank Allah for the strength to complete this project and express gratitude to Dr. Mohammed El-Said for his guidance. Special thanks to the open-source community for tools like OpenCV, EasyOCR, and YOLO.
References

Ultralytics YOLO Documentation
Tesseract OCR
Real-time Egyptian License Plate Detection and Recognition
EALPR System GitHub

