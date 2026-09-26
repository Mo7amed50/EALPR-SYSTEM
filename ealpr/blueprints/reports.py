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

from models import Visitor
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
        col = Visitor._get_collection()
        docs = list(col.find({}, {
            "status": 1,
            "entry_datetime_utc": 1,
            "entry_time": 1,
            "responsible_department": 1,
        }))

        total_visitors = len(docs)
        authorized_visitors = sum(1 for d in docs if d.get("status") == "authorized")
        unauthorized_visitors = sum(1 for d in docs if d.get("status") == "unauthorized")
        pending_visitors = sum(1 for d in docs if d.get("status") == "pending")

        end_date = datetime.utcnow()
        month_start = end_date - timedelta(days=30)

        # Last 7 days daily breakdown
        daily_traffic = []
        daily_status = []
        for i in range(7):
            day = end_date - timedelta(days=i)
            d_str = day.strftime("%Y-%m-%d")
            day_visitors = [
                d for d in docs
                if d.get("entry_datetime_utc") and d["entry_datetime_utc"].strftime("%Y-%m-%d") == d_str
            ]
            auth = sum(1 for d in day_visitors if d.get("status") == "authorized")
            unauth = sum(1 for d in day_visitors if d.get("status") == "unauthorized")
            pending = sum(1 for d in day_visitors if d.get("status") == "pending")

            daily_traffic.append({"date": d_str, "day": day.strftime("%a"), "count": len(day_visitors)})
            daily_status.append({"date": d_str, "authorized": auth, "unauthorized": unauth, "pending": pending})

        daily_traffic.reverse()
        daily_status.reverse()

        # Peak hours (last 30 days)
        month_docs = [
            d for d in docs
            if d.get("entry_datetime_utc") and d["entry_datetime_utc"] >= month_start
        ]
        hour_counts = {}
        for hour in range(24):
            hour_prefix = f"{hour:02d}:"
            h_docs = [
                d for d in month_docs
                if (d.get("entry_time") and str(d["entry_time"]).startswith(hour_prefix)) or
                   (d.get("entry_datetime_utc") and d["entry_datetime_utc"].hour == hour)
            ]
            hour_counts[f"{hour:02d}:00"] = {
                "total": len(h_docs),
                "authorized": sum(1 for d in h_docs if d.get("status") == "authorized"),
                "unauthorized": sum(1 for d in h_docs if d.get("status") == "unauthorized"),
                "pending": sum(1 for d in h_docs if d.get("status") == "pending"),
            }

        # Department stats
        dept_stats = {}
        for d in docs:
            dept = d.get("responsible_department")
            if dept:
                if dept not in dept_stats:
                    dept_stats[dept] = {"total": 0, "authorized": 0, "unauthorized": 0, "pending": 0}
                dept_stats[dept]["total"] += 1
                st = d.get("status")
                if st in dept_stats[dept]:
                    dept_stats[dept][st] += 1

        return jsonify({
            "success": True,
            "summary": {
                "total_visitors": total_visitors,
                "authorized_visitors": authorized_visitors,
                "unauthorized_visitors": unauthorized_visitors,
                "pending_visitors": pending_visitors,
                "authorization_rate": (authorized_visitors / total_visitors * 100) if total_visitors > 0 else 0,
            },
            "daily_traffic": daily_traffic,
            "daily_status": daily_status,
            "peak_hours": hour_counts,
            "department_stats": dept_stats,
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

        col = Visitor._get_collection()
        docs = list(col.find({}, {
            "status": 1,
            "entry_datetime_utc": 1,
            "entry_time": 1,
            "responsible_department": 1,
        }))

        # Summary sheet
        ws_summary = wb.active
        ws_summary.title = "Summary"
        ws_summary.append(["Metric", "Value"])

        total_visitors = len(docs)
        authorized_visitors = sum(1 for d in docs if d.get("status") == "authorized")
        unauthorized_visitors = sum(1 for d in docs if d.get("status") == "unauthorized")
        pending_visitors = sum(1 for d in docs if d.get("status") == "pending")

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
        end_date = datetime.utcnow()
        for i in range(30):
            day = end_date - timedelta(days=i)
            d_str = day.strftime("%Y-%m-%d")
            day_visitors = [
                d for d in docs
                if d.get("entry_datetime_utc") and d["entry_datetime_utc"].strftime("%Y-%m-%d") == d_str
            ]
            auth = sum(1 for d in day_visitors if d.get("status") == "authorized")
            unauth = sum(1 for d in day_visitors if d.get("status") == "unauthorized")
            pending = sum(1 for d in day_visitors if d.get("status") == "pending")
            ws_daily.append([
                d_str,
                day.strftime("%a"),
                len(day_visitors),
                auth,
                unauth,
                pending,
            ])
        for cell in ws_daily[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B", "C", "D", "E", "F"]:
            ws_daily.column_dimensions[col_name].width = 15

        # Hourly traffic sheet
        ws_hourly = wb.create_sheet("Hourly Traffic")
        ws_hourly.append(["Hour", "Total", "Authorized", "Unauthorized", "Pending"])
        month_start = end_date - timedelta(days=30)
        month_docs = [
            d for d in docs
            if d.get("entry_datetime_utc") and d["entry_datetime_utc"] >= month_start
        ]
        for hour in range(24):
            hour_str = f"{hour:02d}:00"
            hour_prefix = f"{hour:02d}:"
            h_docs = [
                d for d in month_docs
                if (d.get("entry_time") and str(d["entry_time"]).startswith(hour_prefix)) or
                   (d.get("entry_datetime_utc") and d["entry_datetime_utc"].hour == hour)
            ]
            ws_hourly.append([
                hour_str,
                len(h_docs),
                sum(1 for d in h_docs if d.get("status") == "authorized"),
                sum(1 for d in h_docs if d.get("status") == "unauthorized"),
                sum(1 for d in h_docs if d.get("status") == "pending"),
            ])
        for cell in ws_hourly[1]:
            cell.font = Font(bold=True)
        for col_name in ["A", "B", "C", "D", "E"]:
            ws_hourly.column_dimensions[col_name].width = 15

        # Department statistics sheet
        ws_dept = wb.create_sheet("Department Statistics")
        ws_dept.append(["Department", "Total Visitors", "Authorized", "Unauthorized", "Pending", "Authorization Rate"])
        dept_stats = {}
        for d in docs:
            dept = d.get("responsible_department")
            if dept:
                if dept not in dept_stats:
                    dept_stats[dept] = {"total": 0, "authorized": 0, "unauthorized": 0, "pending": 0}
                dept_stats[dept]["total"] += 1
                st = d.get("status")
                if st in dept_stats[dept]:
                    dept_stats[dept][st] += 1

        for dept, stats in sorted(dept_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            tot = stats["total"]
            aut = stats["authorized"]
            ws_dept.append([dept, tot, aut, stats["unauthorized"], stats["pending"], f"{(aut / tot * 100) if tot > 0 else 0:.1f}%"])
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

        visitors = Visitor.objects(
            entry_datetime_utc__gte=start_date,
            entry_datetime_utc__lte=end_date,
        ).order_by("entry_datetime_utc")

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
