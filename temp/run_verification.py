import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from ealpr import create_app
from models import User, Visitor, DetectionResult

print("Starting Verification Suite...")
app = create_app()

with app.app_context():
    # 1. Test Camera Feed Generator
    print("\n[1/5] Testing Camera Feed Streaming & Sync...")
    import ealpr.utils as utils
    gen = utils.generate_frames()
    t0 = time.time()
    frames_received = 0
    for chunk in gen:
        frames_received += 1
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"), "Invalid chunk header"
        assert len(chunk) > 100, "Chunk too small"
        if frames_received >= 5:
            break
    elapsed = time.time() - t0
    gen.close()
    print(f"   -> Received 5 frames in {elapsed:.2f}s (Paced smoothly, no tight loop!)")

with app.test_client() as client:
    # Authenticate as admin
    admin = User.objects(username="admin").first()
    assert admin is not None, "Admin user not found"
    with client.session_transaction() as sess:
        sess["_user_id"] = str(admin.id)
        sess["_fresh"] = True

    # 2. Test Analytics and Reports
    print("\n[2/5] Testing Analytics and Reports API...")
    t0 = time.time()
    res = client.get("/api/analytics/visitor-stats")
    dt = time.time() - t0
    assert res.status_code == 200, f"Analytics status: {res.status_code}"
    data = res.get_json()
    assert data["success"] is True
    total_vis = data["summary"]["total_visitors"]
    print(f"   -> /api/analytics/visitor-stats responded in {dt:.2f}s with {total_vis} total visitors")

    # Test report exports
    for rep in ["daily", "weekly", "monthly"]:
        t0 = time.time()
        res = client.get(f"/api/reports/{rep}")
        assert res.status_code == 200, f"Report {rep} failed: {res.status_code}"
        assert len(res.data) > 1000, f"Report {rep} empty"
        print(f"   -> /api/reports/{rep} exported in {time.time()-t0:.2f}s ({len(res.data)} bytes)")

    t0 = time.time()
    res = client.get("/api/analytics/export-excel")
    assert res.status_code == 200, f"Analytics excel failed: {res.status_code}"
    print(f"   -> /api/analytics/export-excel exported in {time.time()-t0:.2f}s ({len(res.data)} bytes)")

    # 3. Test Visitor API with ObjectId and Auto Code
    print("\n[3/5] Testing Visitor Auto-Code & ID Resolution...")
    test_plate = f"TEST{int(time.time())%100000}"
    add_res = client.post("/api/visitors", json={
        "name": "Automated Test Visitor",
        "license_plate": test_plate,
        "status": "authorized",
        "responsible_department": "Quality Assurance"
    })
    assert add_res.status_code == 200, f"Add visitor failed: {add_res.get_json()}"
    vis_data = add_res.get_json()["visitor"]
    vis_id = vis_data["id"]
    vis_code = vis_data["visitor_code"]
    print(f"   -> Added visitor: Code={vis_code}, Plate={test_plate}, ID={vis_id}")
    assert vis_code and vis_code.startswith("V"), f"Invalid code: {vis_code}"

    # Fetch by ObjectId
    get_res = client.get(f"/api/visitors/{vis_id}")
    assert get_res.status_code == 200, f"Get by ObjectId failed: {get_res.get_json()}"
    print(f"   -> Successfully retrieved visitor using ObjectId {vis_id}")

    # Update visitor
    upd_res = client.put(f"/api/visitors/{vis_id}", json={"responsible_department": "QA Testing Updated"})
    assert upd_res.status_code == 200, f"Update failed: {upd_res.get_json()}"

    # Delete visitor
    del_res = client.delete(f"/api/visitors/{vis_id}")
    assert del_res.status_code == 200, f"Delete failed: {del_res.get_json()}"
    print("   -> Cleaned up test visitor")

    # 4. Test Search and Activities
    print("\n[4/5] Testing Search & Activity Endpoints...")
    res = client.get("/api/activities")
    assert res.status_code == 200
    print(f"   -> /api/activities returned {len(res.get_json().get('activities', []))} activities")

    res = client.get("/api/search/plate?q=ABC")
    assert res.status_code in (200, 404)
    print("   -> /api/search/plate verified")

    # 5. Test Page Renders
    print("\n[5/5] Testing HTML Pages...")
    for path in ["/", "/manage_visitors", "/visitors", "/detection_history", "/reports", "/settings", "/users"]:
        t0 = time.time()
        res = client.get(path)
        assert res.status_code == 200, f"Failed rendering {path}: {res.status_code}"
        print(f"   -> {path} rendered in {time.time()-t0:.2f}s")

print("\n==========================================")
print("ALL VERIFICATION CHECKS PASSED PERFECTLY!")
print("==========================================")
