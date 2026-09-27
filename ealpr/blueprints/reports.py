"""
blueprints/reports.py — Report generation and analytics routes.

Routes:
    GET /reports
    GET /api/analytics/visitor-stats
    GET /api/analytics/export-excel
    GET /api/reports/<type>         (type: daily | weekly | monthly)
"""
import io
from datetime import datetime, timedelta

from flask import Blueprint, render_template, request, jsonify, Response, flash, redirect, url_for
from flask_login import login_required
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

from ealpr import extensions
from ealpr.decorators import db_required
from ealpr.repositories import ReportRepository, VisitorRepository
from ealpr.utils import utc_to_cairo

reports_bp = Blueprint("reports", __name__)


@reports_bp.before_request
def check_db_available():
    if not extensions.DB_ENABLED:
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({
                "success": False,
                "message": "Database unavailable. Please try again later."
            }), 503
        flash("Database is offline. Reports are currently unavailable.", "warning")
        return redirect(url_for("main.index"))


@reports_bp.route("/reports")
@login_required
@db_required
def reports():
    return render_template("reports.html")


# ── Visitor statistics (JSON) ────────────────────────────────────────────────

@reports_bp.route("/api/analytics/visitor-stats")
@login_required
@db_required
def get_visitor_stats():
    try:
        # Phase 2 / Block C1. This handler used to pull all 1023 visitors into
        # Python and then walk that list about forty times to build the summary,
        # the 7-day breakdown, the 24 peak-hour buckets and the department
        # table. ReportRepository.visitor_stats computes the same figures with
        # four MongoDB aggregation facets in a single round trip, and the
        # response body below is byte-for-byte the same.
        stats = ReportRepository.visitor_stats(datetime.utcnow())

        return jsonify({
            "success": True,
            "summary": {
                "total_visitors": stats["total_visitors"],
                "authorized_visitors": stats["authorized_visitors"],
                "unauthorized_visitors": stats["unauthorized_visitors"],
                "pending_visitors": stats["pending_visitors"],
                "authorization_rate": (
                    stats["authorized_visitors"] / stats["total_visitors"] * 100
                ) if stats["total_visitors"] > 0 else 0,
            },
            # The original built these two lists newest-first and then reversed
            # them, so the endpoint always answered oldest-first.
            "daily_traffic": [
                {
                    "date": row.date,
                    "day": row.day,
                    "count": row.total,
                }
                for row in reversed(stats["daily_traffic"])
            ],
            "daily_status": [
                {
                    "date": row.date,
                    "authorized": row.authorized,
                    "unauthorized": row.unauthorized,
                    "pending": row.pending,
                }
                for row in reversed(stats["daily_status"])
            ],
            "peak_hours": stats["peak_hours"],
            "department_stats": stats["department_stats"],
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── Export analytics to Excel ────────────────────────────────────────────────

@reports_bp.route("/api/analytics/export-excel")
@login_required
@db_required
def export_analytics_excel():
    try:
        wb = Workbook()

        # Phase 2 / Block C1: the four aggregation blocks below (summary, 30-day
        # daily, 24 peak hours, departments) each used to re-walk the full
        # unprojected document list in Python. They now come from
        # ReportRepository.excel_tables, which computes the same tables with
        # MongoDB aggregations. The emitted cells are unchanged.
        # One instant drives both the aggregation and the filename, so a
        # request that straddles midnight cannot disagree with itself.
        end_date = datetime.utcnow()
        tables = ReportRepository.excel_tables(end_date)
        total_visitors = tables.summary["total_visitors"]
        authorized_visitors = tables.summary["authorized"]
        unauthorized_visitors = tables.summary["unauthorized"]
        pending_visitors = tables.summary["pending"]

        # Summary sheet
        ws_summary = wb.active
        ws_summary.title = "Summary"
        ws_summary.append(["Metric", "Value"])

        for row in [
            ["Total Visitors", total_visitors],
            ["Authorized Visitors", authorized_visitors],
            ["Unauthorized Visitors", unauthorized_visitors],
            ["Pending Visitors", pending_visitors],
            ["Authorization Rate", f"{(authorized_visitors / total_visitors * 100) if total_visitors > 0 else 0:.1f}%"],
        ]:
            ws_summary.append(row)
        for cell in ws_summary[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B"]:
            ws_summary.column_dimensions[col_name].width = 20

        # Daily traffic sheet (last 30 days)
        ws_daily = wb.create_sheet("Daily Traffic")
        ws_daily.append(["Date", "Day", "Total", "Authorized", "Unauthorized", "Pending"])
        for row in tables.daily:
            ws_daily.append([
                row.date,
                row.day,
                row.total,
                row.authorized,
                row.unauthorized,
                row.pending,
            ])
        for cell in ws_daily[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B", "C", "D", "E", "F"]:
            ws_daily.column_dimensions[col_name].width = 15

        # Hourly traffic sheet
        ws_hourly = wb.create_sheet("Hourly Traffic")
        ws_hourly.append(["Hour", "Total", "Authorized", "Unauthorized", "Pending"])
        for row in tables.hourly.items():
            ws_hourly.append([
                row[0],
                row[1]["total"],
                row[1]["authorized"],
                row[1]["unauthorized"],
                row[1]["pending"],
            ])
        for cell in ws_hourly[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B", "C", "D", "E"]:
            ws_hourly.column_dimensions[col_name].width = 15

        # Department statistics sheet
        ws_dept = wb.create_sheet("Department Statistics")
        ws_dept.append(["Department", "Total Visitors", "Authorized", "Unauthorized", "Pending", "Authorization Rate"])
        for row in tables.departments_sorted_by_total:
            tot = row["total"]
            aut = row["authorized"]
            ws_dept.append([row["department"], tot, aut, row["unauthorized"], row["pending"], f"{(aut / tot * 100) if tot > 0 else 0:.1f}%"])
        for cell in ws_dept[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B", "C", "D", "E", "F"]:
            ws_dept.column_dimensions[col_name].width = 20

        excel_file = io.BytesIO()
        wb.save(excel_file)
        excel_file.seek(0)

        return Response(
            excel_file.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename=visitor_analytics_{end_date.strftime("%Y%m%d")}.xlsx'},
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ── Styled period reports ────────────────────────────────────────────────────

@reports_bp.route("/api/reports/<type>")
@login_required
@db_required
def export_report(type):
    if type not in ["daily", "weekly", "monthly"]:
        return jsonify({"success": False, "message": "Invalid report type"}), 400

    excel_file = None
    try:
        wb = Workbook()
        ws = wb.active

        title_font = Font(bold=True, size=14, color="366092")
        header_font = Font(bold=True, color="FFFFFF")
        regular_font = Font(size=10)
        thin_border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin"),
        )

        end_date = datetime.utcnow()
        if type == "daily":
            start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
            title = f"Daily Visitor Report - {end_date.strftime('%Y-%m-%d')}"
        elif type == "weekly":
            start_date = end_date - timedelta(days=7)
            title = f"Weekly Visitor Report ({start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')})"
        else:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            title = f"Monthly Visitor Report - {end_date.strftime('%B %Y')}"

        # Phase 2 / Block C1: projected to the seven columns this workbook
        # writes, instead of loading whole visitor documents.
        visitors = VisitorRepository.list_for_period_report(start_date, end_date)

        ws.merge_cells("A1:G1")
        ws["A1"] = title
        ws["A1"].font = title_font
        ws["A1"].alignment = Alignment(horizontal="center")

        headers = ["Visitor Name", "ID", "License Plate", "Date", "Time", "Status", "Department"]
        ws.append([""] * len(headers))
        ws.append(headers)

        header_row = 3
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=header_row, column=col_idx)
            cell.font = header_font
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.border = thin_border
            cell.alignment = Alignment(horizontal="center", vertical="center")

        for visitor in visitors:
            entry_time = utc_to_cairo(visitor.entry_datetime_utc) if visitor.entry_datetime_utc else ""
            if " " in entry_time:
                date_str, time_str = entry_time.split(" ", 1)
            else:
                date_str, time_str = entry_time or "N/A", "N/A"

            status_str = (visitor.status or "pending").title()
            row = [
                visitor.name or "N/A",
                visitor.visitor_code or (f"V{visitor.visitor_id:03d}" if visitor.visitor_id else "N/A"),
                visitor.license_plate or "N/A",
                date_str,
                time_str,
                status_str,
                visitor.responsible_department or "N/A",
            ]
            ws.append(row)
            current_row = ws.max_row
            for col_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=current_row, column=col_idx)
                cell.font = regular_font
                cell.border = thin_border
                cell.alignment = Alignment(horizontal="left")
                if current_row % 2 == 0:
                    cell.fill = PatternFill(start_color="F5F5F5", end_color="F5F5F5", fill_type="solid")

        column_widths = {"A": 30, "B": 15, "C": 15, "D": 12, "E": 10, "F": 15, "G": 20}
        for col_name, width in column_widths.items():
            ws.column_dimensions[col_name].width = width

        if ws.max_row >= 3:
            ws.auto_filter.ref = f"A3:G{ws.max_row}"

        excel_file = io.BytesIO()
        wb.save(excel_file)
        excel_file.seek(0)

        filename = f"{type}_report_{end_date.strftime('%Y%m%d')}.xlsx"
        return Response(
            excel_file.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        if excel_file:
            try:
                excel_file.close()
            except Exception:
                pass
        return jsonify({"success": False, "message": str(e)}), 500
