# EALPR V2 — MASTER PROJECT SPECIFICATION

**Project:** EALPR — Egyptian Automatic License Plate Recognition System  
**Version:** 2.0  
**Status:** Master Specification / Source of Truth

## 1. Vision

EALPR V2 is a complete AI-powered vehicle and license-plate recognition platform combining a modern responsive UI, Flask backend, REST APIs, MongoDB, YOLO plate detection, OCR, Arabic text processing, visitor management, image/video detection, live cameras, history, reports, users, notifications, audit logs, security, testing, and documentation.

The system must feel like a modern AI/security operations platform rather than a basic CRUD application.

## 2. Core Goals

- Detect Egyptian license plates from images.
- Process videos and live cameras.
- Read plate text with OCR.
- Normalize Arabic text correctly.
- Match plates with registered visitors.
- Determine authorized/unauthorized status.
- Store and search detections.
- Manage visitors, cameras, users, and permissions.
- Provide dashboard analytics and reports.
- Export CSV/Excel/PDF.
- Provide secure authentication and authorization.
- Provide notifications and audit logs.
- Maintain fast, responsive UI with no freezing.

## 3. Critical Performance Requirement

The existing reported freezing problem must be treated as a top-priority architectural issue.

Normal page navigation MUST NOT:

- run YOLO;
- run OCR;
- load all historical images;
- load all history records unnecessarily;
- embed huge Base64 images;
- start camera loops;
- execute blocking loops;
- initialize heavy models repeatedly.

Historical records must display stored results. Viewing a historical detection must never rerun inference.

Target flow:

```text
User opens page
  -> Fast UI response
  -> Lightweight API/database request
  -> Lazy/async loading
  -> AI only when explicitly requested
```

## 4. Target Architecture

```text
                    EALPR V2
                       |
          +------------+------------+
          |                         |
       FRONTEND                  BACKEND
          |                         |
   HTML/CSS/JS                 Flask REST API
          |                         |
    UI Components             Services Layer
          |                    /                 API Calls             AI          Data
                            /   \                                     YOLO  OCR       MongoDB
          |
   Image/Video/Camera Storage
```

## 5. Recommended Structure

```text
EALPR/
├── app.py
├── config.py
├── requirements.txt
├── .env
├── .env.example
├── PROJECT_SPEC.md
├── README.md
├── API_DOCUMENTATION.md
├── DATABASE_SCHEMA.md
├── DEPLOYMENT.md
├── app/
│   ├── auth/
│   ├── dashboard/
│   ├── detection/
│   ├── history/
│   ├── visitors/
│   ├── cameras/
│   ├── reports/
│   ├── users/
│   ├── notifications/
│   ├── settings/
│   ├── services/
│   │   ├── yolo_service.py
│   │   ├── ocr_service.py
│   │   ├── detection_service.py
│   │   ├── image_service.py
│   │   ├── video_service.py
│   │   ├── camera_service.py
│   │   ├── report_service.py
│   │   └── notification_service.py
│   ├── repositories/
│   ├── models/
│   ├── utils/
│   ├── templates/
│   │   ├── base.html
│   │   ├── auth/
│   │   ├── dashboard/
│   │   ├── detection/
│   │   ├── history/
│   │   ├── visitors/
│   │   ├── cameras/
│   │   ├── reports/
│   │   ├── users/
│   │   ├── notifications/
│   │   ├── settings/
│   │   ├── errors/
│   │   └── components/
│   └── static/
│       ├── css/
│       ├── js/
│       ├── images/
│       └── icons/
├── ai_models/
│   ├── yolo11m_car_plate/
│   └── yolo_car_plate/
├── storage/
│   ├── uploads/
│   ├── detections/
│   ├── thumbnails/
│   └── exports/
├── tests/
└── scripts/
```

## 6. UI/UX Design System

Primary references:

- shadcn/ui
- Tailwind CSS
- Radix UI accessibility and interaction principles
- Material Design 3
- Supabase Design System
- dashboardcn

Do not blindly copy another dashboard. Create an EALPR-specific visual identity.

Design direction: **Modern AI / Security Operations Dashboard**.

The interface should communicate reliability, security, technology, clarity, speed, and professionalism.

## 7. Main Layout

```text
+------------------------------------------------------+
| EALPR                         Notifications  Profile |
+----------------+-------------------------------------+
| Dashboard      |                                     |
| Detection      |             MAIN CONTENT             |
| History        |                                     |
| Visitors       |                                     |
| Cameras        |                                     |
| Reports        |                                     |
| Users          |                                     |
| Settings       |                                     |
| Logout         |                                     |
+----------------+-------------------------------------+
```

Responsive behavior:

- Desktop: expanded sidebar.
- Laptop: compact/expanded sidebar.
- Tablet: collapsible sidebar.
- Mobile: drawer navigation.

## 8. Templates

Templates are presentation only. Users never interact with template filenames directly.

Recommended:

```text
templates/
├── base.html
├── auth/login.html
├── auth/signup.html
├── dashboard/dashboard.html
├── detection/detection.html
├── detection/image_result.html
├── detection/video_result.html
├── history/history.html
├── history/detection_details.html
├── visitors/visitors.html
├── visitors/add.html
├── visitors/edit.html
├── visitors/details.html
├── cameras/cameras.html
├── cameras/add_camera.html
├── cameras/camera_details.html
├── reports/reports.html
├── users/users.html
├── users/user_details.html
├── notifications/notifications.html
├── settings/settings.html
├── errors/404.html
├── errors/500.html
└── components/
    ├── navbar.html
    ├── sidebar.html
    ├── modal.html
    ├── table.html
    ├── pagination.html
    ├── toast.html
    ├── loading.html
    ├── skeleton.html
    ├── empty_state.html
    └── confirmation.html
```

`base.html` contains the global shell. Pages extend it. Shared components must be reused.

## 9. Authentication

### Sign In

- Username/email
- Password
- Validation
- Secure session
- Loading and error states
- Optional secure forgot-password flow

### Sign Up

- Full name
- Username
- Email
- Password
- Confirm password

Rules:

- Validate inputs.
- Prevent duplicates.
- Hash passwords securely.
- Public signup creates a normal User.
- Public signup never creates an Admin.

### Logout

Destroy/expire the session and prevent access to protected pages.

## 10. Roles

### Admin

Dashboard, Detection, History, Visitors, Cameras, Reports, Users, Settings, Audit Logs.

### Operator

Dashboard, Detection, History, Visitors, Cameras, Reports according to permissions.

### Viewer

Dashboard, History, Reports.

Backend authorization must enforce permissions even if a UI control is hidden.

## 11. Dashboard

KPI cards:

- Total Detections
- Today's Detections
- Authorized Vehicles
- Unauthorized Vehicles
- Active Cameras
- Registered Visitors

Charts:

- Detection trend
- Authorized vs Unauthorized
- Detection by Camera
- Detection by Source

Recent detections:

- Thumbnail
- Plate
- Status
- Camera
- Source
- Date/Time
- Action

Dashboard must use stored data only and never run YOLO/OCR.

## 12. Image Detection

Flow:

```text
Upload
 -> Validate
 -> Preview
 -> User clicks Detect
 -> YOLO
 -> Plate Crop
 -> OCR
 -> Arabic Processing
 -> Visitor Lookup
 -> Result
 -> Optional Save
```

UI must support:

- Drag/drop
- File picker
- Preview
- Supported format/size feedback
- Loading/progress
- Error states
- Result view
- New Detection

## 13. Detection Result

Display:

- Original image
- Bounding box
- Plate crop
- Plate text
- YOLO confidence
- OCR confidence
- Visitor
- Status
- Camera/source
- Timestamp

Example:

```text
Plate: ABC123
Detection Confidence: 99.2%
OCR Confidence: 95.4%
Visitor: Ahmed Mohamed
Status: Authorized
```

## 14. Video Detection

```text
Upload Video
 -> Validate
 -> Frame Sampling
 -> YOLO
 -> Tracking / Temporal Consistency
 -> OCR when necessary
 -> Duplicate Suppression
 -> Results
```

Provide preview, processing state, progress, detection count, unique plates, results, and export.

Do not process every frame unnecessarily.

## 15. Live Cameras

Camera management:

- Add/edit/delete
- Enable/disable
- Start/stop
- View
- Status
- Location
- Stream URL

Processing:

```text
Stream
 -> Frame Sampling
 -> YOLO
 -> Tracking / Temporal Consistency
 -> Selective OCR
 -> Deduplication
 -> Confirmed Detection
 -> MongoDB
```

Camera processing must not block Flask page requests.

## 16. Detection History

Required:

- Search
- Plate filter
- Status filter
- Camera filter
- Source filter
- Date range
- Sorting
- Pagination (20/50/100)

Table:

```text
Image | Plate | Confidence | Status | Camera | Date | Action
```

Clicking View must not reload the whole page unnecessarily, run AI, load every image, or block the browser.

Use thumbnail + async details + modal/lightbox + full image only when requested + spinner + error state + close/Escape.

## 17. Visitors

Fields:

```text
visitor_code
name
license_plate
entry_date
entry_time
authorized
responsible_department
general_department
created_at
updated_at
```

Features:

- Add
- Edit
- Delete
- View
- Search
- Filter
- Plate lookup
- Authorization management

Matching:

```text
Detected Plate
 -> Normalize
 -> Visitor Lookup
 -> Match?
    Yes -> Visitor + Status
    No  -> Unknown/Unauthorized
```

## 18. Reports

Filters:

- Date range
- Plate
- Visitor
- Status
- Camera
- Source

Statistics:

- Total detections
- Authorized
- Unauthorized
- Unique vehicles
- Detections per day
- Detections per camera

Exports:

- CSV
- Excel
- PDF

Reports must query stored data and never rerun YOLO/OCR.

## 19. User Management

Admin features:

- List/add/edit users
- Change role
- Enable/disable
- Reset password
- Delete when appropriate
- Search/filter

## 20. Notifications

Example:

```text
Unauthorized Vehicle Detected
Plate: ABC123
Camera: Gate 2
Time: 10:42
```

Support read/unread, mark as read, timestamp, related detection, and appropriate clearing.

## 21. Global Search

Search relevant entities:

- Visitors
- Detections
- Cameras

Group results by entity type.

## 22. MongoDB

Collections:

```text
users
visitors
detections
cameras
notifications
audit_logs
```

Detection fields:

```text
_id
plate_text
plate_confidence
detection_confidence
image_path
thumbnail_path
source
camera_id
visitor_id
status
created_at
```

Recommended indexes:

```text
detections.plate_text
detections.created_at
detections.camera_id
detections.status
detections.visitor_id
visitors.license_plate
users.username
users.email
```

Actual existing schema must be audited before migration.

## 23. AI Services

### YOLO Service

Responsibilities:

- Model loading/reuse
- Detection
- Bounding boxes
- Confidence
- Safe inference

Known model paths:

```text
yolo11m_car_plate/best.pt
yolo_car_plate/best.pt
```

Do not reload the model for every request.

### OCR Service

Responsibilities:

- Cropped plate input
- OCR
- Confidence
- Arabic processing
- Normalization
- Safe failure handling

### Detection Service

Central orchestration:

```text
Image
 -> YOLO
 -> Crop
 -> OCR
 -> Normalize
 -> Visitor Lookup
 -> Result
```

Routes should call services, not contain the whole AI pipeline.

## 24. Storage

```text
storage/
├── uploads/
├── detections/
├── thumbnails/
└── exports/
```

Use references/metadata in MongoDB instead of unnecessary huge image payloads.

Requirements:

- Secure filenames
- File type validation
- File size limits
- Path traversal protection
- Thumbnails
- Controlled image serving
- Temporary-file cleanup

## 25. API

```text
POST /api/auth/login
POST /api/auth/signup
POST /api/auth/logout

GET  /api/dashboard/stats

POST /api/detection/image
POST /api/detection/video

GET  /api/history
GET  /api/history/<id>

GET    /api/visitors
POST   /api/visitors
GET    /api/visitors/<id>
PUT    /api/visitors/<id>
DELETE /api/visitors/<id>

GET    /api/cameras
POST   /api/cameras
GET    /api/cameras/<id>
PUT    /api/cameras/<id>
DELETE /api/cameras/<id>
POST   /api/cameras/<id>/start
POST   /api/cameras/<id>/stop

GET /api/reports
GET /api/reports/export

GET    /api/users
POST   /api/users
PUT    /api/users/<id>
DELETE /api/users/<id>
```

Use OpenAPI/Swagger documentation.

## 26. UI Components

Reusable components:

- Buttons
- Inputs
- Selects
- Checkbox/radio/switch
- Data tables
- Cards
- Badges
- Dialogs/modals
- Dropdowns
- Tooltips
- Tabs
- Toasts
- Alerts
- Spinners
- Skeletons
- Empty/error states
- Pagination
- Date picker
- File uploader
- Image preview
- Confirmation dialog
- Search bar

Use one consistent component system.

# 27. Animation System

Animation is a required part of EALPR V2.

Goal: polished and modern without causing performance problems.

## Principles

1. Subtle and purposeful.
2. Communicate interaction or state.
3. Never unnecessarily delay the user.
4. Never block JavaScript.
5. Prefer GPU-friendly `transform` and `opacity`.
6. Avoid expensive animations on large DOM trees.
7. Never animate thousands of table rows simultaneously.
8. Respect `prefers-reduced-motion`.
9. Never use animation instead of meaningful loading feedback.

## Global animations

- Page transitions: subtle fade/slide, approximately 150–300ms.
- Sidebar: smooth open/close.
- Navbar: menu fade/scale.
- Cards: small hover elevation/transform.
- Buttons: hover/active/disabled transitions.
- Modals: fade + small scale/translate.
- Toasts: slide/fade.
- Tooltips: small fade/scale.

## Detection animations

Show:

```text
Uploading
 -> Processing Image
 -> Detecting Plate
 -> Reading Plate
 -> Matching Visitor
 -> Completed
```

Use animated progress/status indicators.

A detected plate may use a subtle bounding-box reveal and result-card entrance.

Do not animate the AI computation itself.

## Upload

Support:

- Drag-over animation
- Upload progress
- Preview fade-in
- Remove transition

## History

Use:

- Skeleton loading
- Table/container fade-in
- Modal transition
- Image fade-in
- Status transitions

Do not animate hundreds of rows.

## Cameras

Status indicators:

```text
Online      -> subtle indicator
Connecting  -> controlled pulse
Offline     -> static warning
Processing  -> subtle activity indicator
```

Do not place heavy animation over live video.

## Dashboard

- KPI cards with small initial stagger.
- Charts animate once.
- Recent detections fade in.
- Avoid continuous decorative animation.

## Accessibility

Respect:

```css
@media (prefers-reduced-motion: reduce)
```

Reduce/disable decorative motion while keeping status information visible.

## 28. Performance

Mandatory:

- No heavy inference during normal navigation.
- No repeated model loading.
- No repeated OCR initialization.
- Pagination for large datasets.
- Thumbnail images in lists.
- Lazy-load full images.
- MongoDB indexes.
- Avoid huge HTML/Base64.
- Avoid synchronous blocking requests.
- Avoid long JS loops.
- Avoid excessive DOM manipulation.
- Isolate camera processing.
- Profile before adding infrastructure.

## 29. Security

Implement:

- Secure password hashing
- Authentication
- Authorization
- RBAC
- Session security
- CSRF where applicable
- Input validation
- File validation
- File size limits
- Secure filenames
- Path traversal protection
- Environment variables
- No secrets in frontend
- Safe API errors
- Audit logs
- Appropriate rate limiting

## 30. Error Handling

Friendly UI states:

### 404

```text
Page not found
[ Back to Dashboard ]
```

### 500

```text
Something went wrong.
Please try again.
```

### Detection failure

```text
We could not process this image.
Please verify the image and try again.
```

Never expose stack traces or server filesystem paths to users.

## 31. Loading States

Every potentially slow operation needs feedback:

- Page
- Table
- Image
- Detection
- OCR
- Reports
- Export
- Camera connection

Use skeletons, spinners, progress indicators, and status text.

Never leave the UI apparently unresponsive.

## 32. Testing

### Unit

- YOLO
- OCR
- Arabic normalization
- Visitor matching
- Authentication
- Authorization
- Reports
- File validation

### Integration

```text
Image -> YOLO -> OCR -> Visitor lookup -> MongoDB
```

### UI

Test:

- Login
- Signup
- Dashboard
- Detection
- History
- View modal
- Visitors
- Cameras
- Reports
- Users
- Logout

### Freeze regression

Verify:

- History opens normally.
- View opens normally.
- Modal closes.
- Navigation remains responsive.
- Large history does not freeze.
- Upload remains responsive.
- Reports remain responsive.
- No hidden inference occurs on navigation.

## 33. Logging

Log:

- Authentication events
- API errors
- Detection failures
- OCR failures
- Camera errors
- Report failures
- Security events

Never log plain passwords, secrets, or sensitive tokens.

## 34. Audit Logs

Record important administrative actions:

```text
User
Action
Entity
Timestamp
Result
```

Examples:

```text
Admin created user
Admin disabled user
Operator added visitor
Admin changed camera
```

## 35. Development Phases

### Phase 0 — Backup

- Source backup
- MongoDB backup
- Model backup
- Configuration backup

### Phase 1 — Repository Audit

Inspect all routes, templates, JS/CSS, services, models, DB queries, YOLO/OCR entry points, storage, and cameras.

Do not delete unknown functionality.

### Phase 2 — Architecture

Introduce modular structure.

### Phase 3 — Authentication

Login, signup, logout, roles, permissions.

### Phase 4 — UI Foundation

Base layout, sidebar, navbar, components, responsive behavior, theme, animation system.

### Phase 5 — Detection

YOLO, OCR, detection service.

### Phase 6 — Freeze Fix

History, images, queries, JS, model lifecycle.

### Phase 7 — Visitors

CRUD and matching.

### Phase 8 — Upload

Image/video workflow.

### Phase 9 — Reports

Analytics and exports.

### Phase 10 — Cameras

Live processing.

### Phase 11 — Security

Security review.

### Phase 12 — Testing

Unit, integration, UI, performance.

### Phase 13 — Documentation/Deployment

README, Swagger, database docs, deployment docs.

## 36. AI Coding Agent Rules

The coding agent MUST:

1. Read this file before modifying the project.
2. Treat it as the source of truth.
3. Inspect the existing repository before destructive changes.
4. Preserve working features.
5. Never delete a feature without documenting why.
6. Never replace the entire project blindly.
7. Work incrementally.
8. Test after every major phase.
9. Keep the application runnable.
10. Never run YOLO/OCR during ordinary page rendering.
11. Never load all history at once.
12. Use thumbnails and lazy loading.
13. Reuse AI models.
14. Avoid unnecessary dependencies.
15. Document dependency additions.
16. Keep secrets outside source code.
17. Follow the UI/UX design system.
18. Follow the animation system.
19. Respect accessibility and reduced motion.
20. Never expose filesystem paths.
21. Preserve MongoDB data.
22. Create migrations for schema changes.
23. Update documentation after architectural changes.
24. Never claim completion without testing.

## 37. Git and Backup

Before major changes:

```text
git status
git add .
git commit
```

Recommended branches:

```text
main
develop
feature/auth
feature/ui
feature/detection
feature/history
feature/cameras
feature/reports
```

Never perform destructive refactoring without a recoverable checkpoint.

## 38. Definition of Done

EALPR V2 is complete when:

- Authentication works.
- Signup works.
- Roles and permissions work.
- Dashboard works.
- Image detection works.
- Video detection works if enabled.
- OCR works.
- Arabic processing works.
- Visitor matching works.
- Detection history works.
- History is paginated.
- Images use thumbnails/lazy loading.
- Detection details open without freezing.
- Upload works.
- Reports work.
- Exports work.
- Camera management works.
- Live camera processing works if enabled.
- Users can be managed.
- Notifications work.
- Audit logs work.
- Security checks pass.
- Responsive UI works.
- Animations are consistent.
- Reduced-motion support works.
- No unnecessary inference occurs on navigation.
- Automated tests pass.
- Swagger/OpenAPI exists.
- README is updated.
- Deployment documentation exists.

## 39. Final User Experience

```text
Open EALPR
    ↓
Sign In
    ↓
Dashboard
    ↓
Choose:
    ├── Detect Image
    ├── Detect Video
    ├── Live Camera
    ├── View History
    ├── Manage Visitors
    ├── View Reports
    └── Manage Cameras
```

Every action should provide immediate visual feedback.

The system should feel:

- Fast
- Clear
- Professional
- Modern
- Secure
- Responsive
- AI-powered
- Easy to learn

## 40. Final Target

```text
Existing EALPR
      ↓
Repository Audit
      ↓
Modular Architecture
      ↓
Modern UI/UX
      ↓
Animation System
      ↓
Reliable AI Services
      ↓
Secure APIs
      ↓
Optimized MongoDB
      ↓
Testing
      ↓
Documentation
      ↓
Production-like EALPR V2
```

The project must optimize for:

**Correct Architecture + Good UX + Reliable AI + Fast UI + Security + Maintainability**

Do not add Redis, Celery, microservices, or other infrastructure merely because they are popular. Add infrastructure only when actual workload requirements justify it and the reason is documented.

**This document is the master specification and source of truth for EALPR V2.**
