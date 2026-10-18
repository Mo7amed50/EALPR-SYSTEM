EALPR System - Egyptian Automatic License Plate Recognition
Overview
The EALPR System is a smart, AI-based application designed to automate the recognition of Egyptian vehicle license plates in real time. Developed as a graduation project by students at Helwan University's Faculty of Computers and Artificial Intelligence, this system leverages computer vision, deep learning, and database integration to streamline vehicle identification for applications such as traffic monitoring, parking management, and security systems.
The system captures live video feeds, detects license plates, recognizes Arabic characters and digits, and retrieves associated vehicle or driver information from a database. It is tailored specifically for Egyptian license plates, addressing challenges like mixed Arabic and Western numeral formats and varying plate designs.
Features

Real-Time License Plate Detection: Uses a deep learning model (YOLO) to identify and extract license plates from live video streams.
Character Recognition: Uses a second YOLO model (a character-level detector with 38 Arabic letter/digit classes) to read the plate, after CLAHE contrast enhancement and denoising of the cropped plate.
Database Integration: Matches recognized plates with a MongoDB database to retrieve and display vehicle/driver details.
User-Friendly Interface: Provides live camera feed monitoring, detection history, and visitor management functionalities.
Scalable Design: Supports potential integration with smart gates, parking systems, and mobile applications.

Technologies Used

OpenCV: For video streaming, image processing, and license plate detection.
YOLO (Ultralytics): For deep learning-based license plate detection and character recognition.
MongoDB: For storing and querying vehicle and visitor data.
Python: Core programming language for system implementation.
Flask, Flask-Login and Flask-SocketIO: Web backend, authentication and live detection events.
MongoEngine / PyMongo: Database models for MongoDB Atlas.
Waitress: Production WSGI server.

Installation

Clone the Repository:
git clone https://github.com/Mo7amed50/EALPR-SYSTEM.git
cd EALPR-SYSTEM


Install Dependencies:Ensure Python 3.10+ is installed, then set up a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set Up Environment Variables:Create a .env file in the project root and configure the following:
Copy .env.example to .env and fill in your own values. NEVER commit .env: it is
in .gitignore, and it holds your MongoDB Atlas credentials and SECRET_KEY.

MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/ealpr_db?retryWrites=true&w=majority
MONGODB_DB_NAME=ealpr_db
SECRET_KEY=<long random value, e.g. python -c "import secrets; print(secrets.token_hex(32))">
MODEL_PATH=working/runs/detect/yolo_car_plate/weights/best.pt
OCR_MODEL_PATH=working/runs/detect/yolo11m_car_plate/weights/best.pt
TEMP_DIR=temp

If MongoDB is unavailable, the app will still launch in offline/read-only mode, but most database operations will be disabled.

Model weights:The two .pt files above are stored with Git LFS. After cloning, run "git lfs install" and "git lfs pull" so they are real files and not small pointer stubs.

Initialize MongoDB:Ensure your MongoDB (Atlas) connection works, then run the database initialization script from the project root:
python -m scripts.init_db


Run the Application:Start the Flask server to launch the web interface:
python app.py

Access the application at http://localhost:5000.


Runtime Modes and Production Readiness (Phase 2, Block C5)

Startup command: python app.py

The application runs in one of three runtime modes, selected by EALPR_ENV. The mode is resolved first thing in create_app(), before any Flask object or database connection exists, so a rejected configuration stops startup with a clear error instead of becoming a running service with a bad setting.

development (default)
Permissive local mode. A fresh checkout with no .env boots here. The built-in development fallback for SECRET_KEY and the default admin password are accepted, and EALPR_DEBUG is honoured.

production
Strict mode. Startup refuses to continue unless SECRET_KEY is set to a unique random value (the placeholder your-super-secret-key-change-this and the fallback ealpr-dev-secret are both rejected) and DEFAULT_ADMIN_PASSWORD is set to something other than the known default admin123. Debug must be off, and the process count must be one. The failing variable is named in the error message; its value is never echoed into the message, the logs, or crash output.

test
Deterministic mode used by the test suite. Refuses the debug flag and needs no real secrets, so the suite runs on a clean checkout without configuring credentials.

Runtime variables

EALPR_ENV = development | production | test
    Selects the mode above. Default: development. An unrecognised value is an error.

EALPR_DEBUG = true | false
    Debug/interactive debugger. Default: false. Only honoured when EALPR_ENV=development; any other mode raises RuntimeConfigError at startup. Never enable it in production.

EALPR_RELOADER = true | false
    Werkzeug auto-reloader. Default: false. Must stay false, because the app is single process and the reloader's parent/child pair would run two independent instances.

EALPR_WORKERS = 1
    Must be exactly 1. Default: 1.

Single-process requirement

This application is single process, and that is a correctness requirement rather than a deployment preference. The inference job queue guarantees exactly one consumer, orphaned-job recovery assumes one owner, and live Socket.IO broadcasts rely on the in-process connection registry. Two workers, two containers, or a reloader parent/child pair would each break those guarantees. The only supported startup command is python app.py, with EALPR_WORKERS=1 and EALPR_RELOADER=false.

Socket.IO runs threaded within that single process (the flask-socketio extension with threading async mode), which is why adding processes would require an external message queue that the application does not have.

Deployment status

S2, the production WSGI server, is implemented with **Waitress** (3.0.2). Waitress is a threaded WSGI server that runs in the same single process as the Flask application and matches the existing `flask-socketio` threading async mode. The development entry point remains `python app.py`. The production entry point uses the WSGI callable `app:app` via `waitress-serve`.

**Development:**
```powershell
python app.py
```

**Production (binds to loopback by default):**
```powershell
waitress-serve --listen=127.0.0.1:5000 --threads=50 app:app
```

The initial thread count of `50` is a conservative deployment setting, not a benchmarked value. It should be sized appropriately for the expected concurrent load (Socket.IO long-polling requests can hold threads for extended periods, so pool sizing matters).

**Socket.IO transport:** Under Waitress, the WebSocket upgrade is not supported by the current `simple_websocket` integration. Socket.IO clients will negotiate to **HTTP long-polling**. All `live_detection` events remain delivered; WebSocket is not available under this server configuration.

Waitress provides no application-process supervision; process restart must be handled by an external supervisor (systemd, Docker restart policy, Windows Service, etc.). The `atexit` shutdown path runs on normal termination but may be bypassed on SIGTERM (a limitation unchanged from the previous design; C4 recovery handles in-flight jobs on restart). An appropriately configured reverse proxy (nginx, Caddy) is recommended if external access is required.

Block C5 delivered runtime mode validation (S1), single-process enforcement (S3), shutdown lifecycle handling (S4), production secret validation (S5), startup logging discipline (S6), this documentation (S7), and the production WSGI server selection/configuration (S2). Combined with the single-process requirement above, this application is not fully production-ready (remaining work includes rate limiting, ProxyFix if behind a proxy affecting client IP/scheme, dependency hardening, monitoring, and load testing).

All values shown in .env.example are placeholders. Do not commit real credentials.


Usage

Live Camera Feed: Start the camera to monitor real-time license plate detections.
Detection History: View past detections with timestamps, plate numbers, and visitor details.
Visitor Management: Add new visitors or manage user accounts via the interface.
Search and Reports: Filter detection history by plate number or status, and generate reports.

Project Structure

/app.py: Development entry point (also exposes the WSGI callable app:app for Waitress).
/config.py: Configuration, runtime-mode validation and model paths.
/models.py: MongoEngine models (User, Visitor, DetectionResult, SystemSettings, ...).
/ealpr: Application package (app factory, blueprints, repositories, inference jobs, AI models).
/templates, /static: Web interface.
/scripts: Database utilities. Run them from the project root, e.g. python -m scripts.init_db
/tests: Test suite. See the notes about test databases before running it.
/working/runs/detect: YOLO weights (best.pt) for plate detection and OCR (Git LFS).
/detected_plates, /temp: Created at runtime. Not tracked by git (they contain real vehicle photos).
/GATE0_BACKUP_PLAN.md, /PHASE2_*.md, /PROJECT_SPEC_FINAL.md: Engineering reports and specification.

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
We thank Allah for the strength to complete this project and express gratitude to Dr. Mohammed El-Said for his guidance. Special thanks to the open-source community for tools like OpenCV and YOLO.
References

Ultralytics YOLO Documentation
Real-time Egyptian License Plate Detection and Recognition
EALPR System GitHub

