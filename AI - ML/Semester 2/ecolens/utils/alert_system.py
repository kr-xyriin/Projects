"""
utils/alert_system.py
EcoLens — Pollution Alert & Email Notification System
Sends automated alerts to municipal/panchayat authorities.
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
import json


class PollutionAlertSystem:
    """Handles pollution level detection and authority notification."""

    def __init__(self, config: dict = None):
        """
        config keys:
          smtp_host, smtp_port, sender_email, sender_password,
          area_name, area_type (urban/semi_urban/rural/industrial),
          authority_email, authority_name
        """
        self.config = config or {}
        self.alert_log = []

    def _build_html_email(self, pollution_data: dict, predictions: list,
                           area_name: str, authority_info: dict) -> str:
        """Build a rich HTML email body."""
        score_pct = pollution_data.get("score_pct", "N/A")
        level = pollution_data.get("level", "unknown").upper()
        label = pollution_data.get("label", "Unknown")
        emoji = pollution_data.get("emoji", "⚠️")
        color = pollution_data.get("color", "#E74C3C")
        total_items = pollution_data.get("total_items", 0)
        recyclable_pct = pollution_data.get("recyclable_pct", 0)
        breakdown = pollution_data.get("breakdown", {})
        timestamp = datetime.now().strftime("%d %B %Y, %I:%M %p")

        # Breakdown rows
        breakdown_rows = ""
        icons = {"cardboard": "📦", "glass": "🍾", "metal": "🥫",
                 "paper": "📄", "plastic": "🧴", "trash": "🗑️"}
        for cls, count in sorted(breakdown.items(), key=lambda x: -x[1]):
            pct = round(count / total_items * 100, 1) if total_items else 0
            icon = icons.get(cls, "🔹")
            breakdown_rows += f"""
            <tr>
                <td style="padding:8px 12px; border-bottom:1px solid #eee;">{icon} {cls.title()}</td>
                <td style="padding:8px 12px; border-bottom:1px solid #eee; text-align:center;">{count}</td>
                <td style="padding:8px 12px; border-bottom:1px solid #eee; text-align:center;">{pct}%</td>
            </tr>"""

        html = f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family: 'Segoe UI', Arial, sans-serif; background:#f5f5f5; margin:0; padding:20px;">

<div style="max-width:640px; margin:0 auto; background:#fff; border-radius:12px;
            box-shadow:0 4px 20px rgba(0,0,0,0.1); overflow:hidden;">

  <!-- Header -->
  <div style="background:{color}; padding:30px 30px 20px; color:white;">
    <div style="font-size:40px; margin-bottom:8px;">{emoji}</div>
    <h1 style="margin:0; font-size:24px; font-weight:700;">
      {level} POLLUTION ALERT
    </h1>
    <p style="margin:6px 0 0; opacity:0.9; font-size:15px;">
      EcoLens Waste Intelligence System — Automated Alert
    </p>
  </div>

  <!-- Body -->
  <div style="padding:28px 30px;">

    <p style="color:#555; margin-top:0;">
      Dear <strong>{authority_info.get('name', 'Authority')}</strong>
      ({authority_info.get('role', '')}),
    </p>
    <p style="color:#555;">
      Our AI-powered waste monitoring system has detected <strong>{label}</strong>
      in the area of <strong>{area_name}</strong>.
      Immediate attention may be required based on the analysis below.
    </p>

    <!-- Pollution Score Card -->
    <div style="background:{color}18; border:2px solid {color}; border-radius:10px;
                padding:20px; margin:20px 0; text-align:center;">
      <div style="font-size:48px; font-weight:800; color:{color};">{score_pct}</div>
      <div style="font-size:16px; color:{color}; font-weight:600;">Pollution Index Score</div>
      <div style="font-size:14px; color:#666; margin-top:4px;">{label}</div>
    </div>

    <!-- Stats Row -->
    <div style="display:flex; gap:12px; margin:20px 0;">
      <div style="flex:1; background:#f8f9fa; border-radius:8px; padding:15px; text-align:center;">
        <div style="font-size:28px; font-weight:700; color:#2c3e50;">{total_items}</div>
        <div style="font-size:12px; color:#666; text-transform:uppercase;">Items Detected</div>
      </div>
      <div style="flex:1; background:#f8f9fa; border-radius:8px; padding:15px; text-align:center;">
        <div style="font-size:28px; font-weight:700; color:#27AE60;">{recyclable_pct}%</div>
        <div style="font-size:12px; color:#666; text-transform:uppercase;">Recyclable</div>
      </div>
      <div style="flex:1; background:#f8f9fa; border-radius:8px; padding:15px; text-align:center;">
        <div style="font-size:28px; font-weight:700; color:#E74C3C;">
          {round(100 - recyclable_pct)}%
        </div>
        <div style="font-size:12px; color:#666; text-transform:uppercase;">Non-Recyclable</div>
      </div>
    </div>

    <!-- Waste Breakdown Table -->
    <h3 style="color:#2c3e50; margin-bottom:8px;">Waste Type Breakdown</h3>
    <table style="width:100%; border-collapse:collapse; font-size:14px;">
      <thead>
        <tr style="background:#f0f0f0;">
          <th style="padding:10px 12px; text-align:left;">Type</th>
          <th style="padding:10px 12px; text-align:center;">Count</th>
          <th style="padding:10px 12px; text-align:center;">Share</th>
        </tr>
      </thead>
      <tbody>{breakdown_rows}</tbody>
    </table>

    <!-- Recommended Actions -->
    <div style="background:#FFF3CD; border-left:4px solid #F39C12;
                border-radius:4px; padding:15px; margin:20px 0;">
      <h4 style="margin:0 0 8px; color:#856404;">⚡ Recommended Actions</h4>
      <ul style="margin:0; padding-left:20px; color:#555; font-size:14px;">
        <li>Deploy sanitation team to {area_name} immediately</li>
        <li>Identify and address primary waste sources</li>
        <li>Schedule waste collection drive within 48 hours</li>
        <li>Consider awareness campaign for waste segregation</li>
        {'<li style="color:#E74C3C; font-weight:600;">URGENT: Escalate to Pollution Control Board</li>' if level == "CRITICAL" else ''}
      </ul>
    </div>

    <!-- Location & Timestamp -->
    <div style="background:#f8f9fa; border-radius:8px; padding:15px; font-size:13px; color:#666;">
      <div>📍 <strong>Area:</strong> {area_name}</div>
      <div>🕐 <strong>Detected at:</strong> {timestamp}</div>
      <div>🤖 <strong>System:</strong> EcoLens v1.0 — AI Waste Classification</div>
    </div>

  </div>

  <!-- Footer -->
  <div style="background:#2c3e50; padding:15px 30px; color:#aaa; font-size:12px; text-align:center;">
    This is an automated alert from EcoLens Waste Intelligence System. <br>
    Please do not reply to this email. Contact your system administrator for queries.
  </div>

</div>
</body>
</html>"""
        return html

    def send_alert(self, pollution_data: dict, predictions: list,
                   recipient_email: str, recipient_name: str,
                   recipient_role: str, area_name: str) -> dict:
        """Send pollution alert email."""
        sender_email = self.config.get("sender_email")
        sender_password = self.config.get("sender_password")
        smtp_host = self.config.get("smtp_host", "smtp.gmail.com")
        smtp_port = int(self.config.get("smtp_port", 587))

        if not sender_email or not sender_password:
            return {
                "success": False,
                "error": "Email credentials not configured. Set SENDER_EMAIL and SENDER_PASSWORD in .env"
            }

        level = pollution_data.get("level", "unknown").upper()
        score_pct = pollution_data.get("score_pct", "N/A")
        timestamp = datetime.now().strftime("%d %b %Y %H:%M")

        subject = (
            f"🚨 [{level} ALERT] Pollution Detected in {area_name} — "
            f"Score: {score_pct} | {timestamp}"
        )

        authority_info = {"name": recipient_name, "role": recipient_role}
        html_body = self._build_html_email(pollution_data, predictions, area_name, authority_info)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"EcoLens Alert System <{sender_email}>"
        msg["To"] = recipient_email

        # Plain text fallback
        plain = (
            f"EcoLens Pollution Alert\n\n"
            f"Dear {recipient_name},\n\n"
            f"Pollution Level: {level}\n"
            f"Pollution Score: {score_pct}\n"
            f"Area: {area_name}\n"
            f"Items Analyzed: {pollution_data.get('total_items', 0)}\n\n"
            f"Please take immediate action.\n\n"
            f"— EcoLens Waste Intelligence System"
        )

        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, msg.as_string())

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "recipient": recipient_email,
                "area": area_name,
                "level": level,
                "score": pollution_data.get("score"),
                "success": True
            }
            self.alert_log.append(log_entry)
            self._save_log(log_entry)

            return {"success": True, "message": f"Alert sent to {recipient_name} ({recipient_email})"}

        except smtplib.SMTPAuthenticationError:
            return {"success": False,
                    "error": "SMTP authentication failed. Check email/password. "
                             "For Gmail: enable App Passwords in your Google account."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _save_log(self, entry: dict):
        """Append alert to log file."""
        log_path = Path("logs/alerts.jsonl")
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_alert_history(self) -> list:
        """Load alert history from log file."""
        log_path = Path("logs/alerts.jsonl")
        if not log_path.exists():
            return []
        records = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except:
                        pass
        return records[-50:]  # Last 50 alerts
