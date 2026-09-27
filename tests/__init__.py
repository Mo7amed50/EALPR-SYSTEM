"""Test package for EALPR.

The project has no third-party test runner installed, so these tests use the
standard-library ``unittest`` module and are run with:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Every test here is **read-only**. The suites talk to the real MongoDB
instance, so they assert on query shape, projections and aggregation results
rather than writing fixtures, and they never create, update or delete a
Visitor or DetectionResult.
"""
