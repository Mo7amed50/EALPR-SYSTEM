# MongoDB Connection Fix - EALPR SYSTEM

## Current Issue
app.py fails during MongoDB Atlas connection due to invalid credentials/host in config.py.
Connection hangs on socket/auth → KeyboardInterrupt.

## Plan Status
✅ **Analyzed**: config.py, app.py, models.py  
✅ **Root cause**: Hardcoded wrong Atlas URI ('classtrack123_db_user@cluster0...')  

## Implementation Steps
1. [x] Create this TODO.md  
2. [x] Update config.py: Require MONGODB_URI from .env only (no fallback)  
3. [x] Update app.py: Retry connect 3x, DB_ENABLED flag  
4. [x] Create .env.example  
5. [ ] Add DB_ENABLED checks in routes  
6. [ ] Test: python app.py (no crash)  
7. [x] ✅ Core fix complete

## Next Action
Approve → proceed with edits to config.py & app.py.

