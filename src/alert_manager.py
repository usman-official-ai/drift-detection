 # src/alert_manager.py
"""
Alert management system for email and Slack notifications.
Sends alerts when data drift is detected.
"""

import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.logger_config import get_logger
from src.exceptions import AlertDeliveryError

logger = get_logger()

class EmailAlert:
    """
    Handles email alerts using SMTP.
    """
    
    def __init__(self):
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.sender = settings.EMAIL_SENDER
        self.password = settings.EMAIL_PASSWORD
        self.recipients = settings.EMAIL_RECIPIENTS
        self.cc = settings.EMAIL_CC or []
        self.bcc = settings.EMAIL_BCC or []
    
    def send(
        self,
        subject: str,
        body: str,
        attachments: Optional[List[Path]] = None,
        html_body: Optional[str] = None
    ) -> bool:
        """
        Send an email alert.
        
        Args:
            subject: Email subject
            body: Plain text email body
            attachments: List of file paths to attach
            html_body: HTML version of the email body
        
        Returns:
            True if sent successfully, False otherwise
        
        Raises:
            AlertDeliveryError: If email sending fails
        """
        if not self.sender or not self.password:
            logger.warning("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.recipients)
            
            if self.cc:
                msg["Cc"] = ", ".join(self.cc)
            if self.bcc:
                msg["Bcc"] = ", ".join(self.bcc)
            
            # Attach body
            if html_body:
                # Attach both plain and HTML versions
                part1 = MIMEText(body, "plain")
                part2 = MIMEText(html_body, "html")
                msg.attach(part1)
                msg.attach(part2)
            else:
                msg.attach(MIMEText(body, "plain"))
            
            # Attach files
            if attachments:
                for file_path in attachments:
                    if file_path.exists():
                        with open(file_path, "rb") as f:
                            part = MIMEApplication(f.read())
                            part.add_header(
                                "Content-Disposition",
                                "attachment",
                                filename=file_path.name
                            )
                            msg.attach(part)
            
            # Send email
            all_recipients = self.recipients + self.cc + self.bcc
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender, self.password)
                server.sendmail(self.sender, all_recipients, msg.as_string())
            
            logger.info(
                "Email alert sent",
                extra={
                    "subject": subject,
                    "recipients": all_recipients
                }
            )
            
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {str(e)}")
            raise AlertDeliveryError(f"SMTP authentication failed: {str(e)}")
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {str(e)}")
            raise AlertDeliveryError(f"SMTP error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            raise AlertDeliveryError(f"Failed to send email: {str(e)}")

class SlackAlert:
    """
    Handles Slack alerts using webhooks.
    """
    
    def __init__(self):
        self.webhook_url = settings.SLACK_WEBHOOK_URL
        self.channel = settings.SLACK_CHANNEL
        self.username = settings.SLACK_USERNAME
        self.icon_emoji = settings.SLACK_ICON_EMOJI
    
    def send(
        self,
        text: str,
        attachments: Optional[List[Dict[str, Any]]] = None,
        blocks: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Send a Slack alert.
        
        Args:
            text: Message text
            attachments: List of attachment objects
            blocks: List of block kit objects
        
        Returns:
            True if sent successfully, False otherwise
        
        Raises:
            AlertDeliveryError: If Slack delivery fails
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            payload = {
                "text": text,
                "username": self.username,
                "icon_emoji": self.icon_emoji
            }
            
            if self.channel:
                payload["channel"] = self.channel
            
            if attachments:
                payload["attachments"] = attachments
            
            if blocks:
                payload["blocks"] = blocks
            
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            response.raise_for_status()
            
            logger.info("Slack alert sent")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            raise AlertDeliveryError(f"Failed to send Slack alert: {str(e)}")

class AlertManager:
    """
    Manages alert delivery through multiple channels.
    """
    
    def __init__(self):
        self.email_alert = EmailAlert()
        self.slack_alert = SlackAlert()
    
    def send_alerts(self, drift_results: Dict[str, Any]) -> None:
        """
        Send alerts through configured channels.
        
        Args:
            drift_results: Drift detection results
        """
        if not drift_results.get("overall_drift", False):
            logger.info("No significant drift detected, skipping alerts")
            return
        
        # Prepare alert content
        subject, body, html_body = self._prepare_alert_content(drift_results)
        slack_message = self._prepare_slack_message(drift_results)
        slack_attachments = self._prepare_slack_attachments(drift_results)
        
        # Send email alert
        if settings.ENABLE_EMAIL_ALERTS:
            try:
                self.email_alert.send(
                    subject=subject,
                    body=body,
                    html_body=html_body
                )
            except AlertDeliveryError as e:
                logger.error(f"Email alert failed: {str(e)}")
        
        # Send Slack alert
        if settings.ENABLE_SLACK_ALERTS:
            try:
                self.slack_alert.send(
                    text=slack_message,
                    attachments=slack_attachments
                )
            except AlertDeliveryError as e:
                logger.error(f"Slack alert failed: {str(e)}")
    
    def _prepare_alert_content(
        self,
        drift_results: Dict[str, Any]
    ) -> tuple:
        """
        Prepare email alert content.
        
        Args:
            drift_results: Drift detection results
        
        Returns:
            Tuple of (subject, body, html_body)
        """
        drift_count = drift_results["drift_count"]
        total_features = drift_results["total_features"]
        drift_percentage = drift_results["drift_percentage"]
        
        subject = f"[Drift Alert] {drift_count} features drifting ({drift_percentage:.1%})"
        
        # Build body
        body_lines = [
            "DATA DRIFT ALERT",
            "=" * 50,
            "",
            f"Timestamp: {drift_results['timestamp']}",
            f"Drift Count: {drift_count} / {total_features}",
            f"Drift Percentage: {drift_percentage:.1%}",
            f"Overall Drift: {'YES' if drift_results['overall_drift'] else 'NO'}",
            "",
            "DETAILED FEATURE ANALYSIS:",
            "-" * 30,
        ]
        
        html_lines = [
            "<html><body>",
            "<h2>Data Drift Alert</h2>",
            "<table border='1'>",
            "<tr><td><b>Timestamp</b></td><td>{}</td></tr>".format(drift_results['timestamp']),
            "<tr><td><b>Drift Count</b></td><td>{}/{}</td></tr>".format(drift_count, total_features),
            "<tr><td><b>Drift Percentage</b></td><td>{:.1%}</td></tr>".format(drift_percentage),
            "<tr><td><b>Overall Drift</b></td><td>{}</td></tr>".format('YES' if drift_results['overall_drift'] else 'NO'),
            "</table>",
            "<h3>Feature Details:</h3>",
            "<ul>"
        ]
        
        for feature, details in drift_results["features"].items():
            if details.get("drift_detected", False):
                body_lines.append(f"\n  {feature} (DRIFT DETECTED):")
                
                if details["type"] == "numerical":
                    body_lines.append(f"    Test: KS Test")
                    body_lines.append(f"    p-value: {details.get('p_value', 'N/A'):.4f}")
                    body_lines.append(f"    Reference Mean: {details.get('reference_mean', 'N/A'):.2f}")
                    body_lines.append(f"    Incoming Mean: {details.get('incoming_mean', 'N/A'):.2f}")
                    
                    html_lines.append(
                        "<li><b>{}</b> (DRIFT) - KS Test p-value: {:.4f}, "
                        "Mean: {:.2f} vs {:.2f}</li>".format(
                            feature,
                            details.get('p_value', 0),
                            details.get('reference_mean', 0),
                            details.get('incoming_mean', 0)
                        )
                    )
                else:
                    body_lines.append(f"    Test: PSI")
                    body_lines.append(f"    PSI: {details.get('psi', 'N/A'):.4f}")
                    
                    html_lines.append(
                        "<li><b>{}</b> (DRIFT) - PSI: {:.4f}</li>".format(
                            feature,
                            details.get('psi', 0)
                        )
                    )
            else:
                body_lines.append(f"\n  {feature} (No drift)")
                html_lines.append("<li>{} (OK)</li>".format(feature))
        
        body_lines.append("\n" + "=" * 50)
        body_lines.append("ACTION REQUIRED:")
        body_lines.append("1. Review the drift report in 'reports' directory")
        body_lines.append("2. Investigate root cause of data changes")
        body_lines.append("3. Consider model retraining if needed")
        
        html_lines.append("</ul>")
        html_lines.append("<p><b>Action Required:</b></p>")
        html_lines.append("<ol>")
        html_lines.append("<li>Review the drift report in 'reports' directory</li>")
        html_lines.append("<li>Investigate root cause of data changes</li>")
        html_lines.append("<li>Consider model retraining if needed</li>")
        html_lines.append("</ol>")
        html_lines.append("</body></html>")
        
        return subject, "\n".join(body_lines), "\n".join(html_lines)
    
    def _prepare_slack_message(self, drift_results: Dict[str, Any]) -> str:
        """
        Prepare Slack message text.
        
        Args:
            drift_results: Drift detection results
        
        Returns:
            Slack message text
        """
        drift_count = drift_results["drift_count"]
        total_features = drift_results["total_features"]
        
        return (
            f"*Data Drift Alert*\n"
            f"Drift Detected: {drift_count}/{total_features} features\n"
            f"Status: {'HIGH' if drift_results['drift_percentage'] > 0.5 else 'MEDIUM'} severity\n"
            f"Time: {drift_results['timestamp']}"
        )
    
    def _prepare_slack_attachments(self, drift_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare Slack attachments.
        
        Args:
            drift_results: Drift detection results
        
        Returns:
            List of Slack attachments
        """
        drifting_features = [
            feature for feature, details in drift_results["features"].items()
            if details.get("drift_detected", False)
        ]
        
        color = "danger" if drift_results["drift_percentage"] > 0.5 else "warning"
        
        attachment = {
            "color": color,
            "fields": [],
            "footer": "Drift Detection System",
            "ts": int(datetime.now().timestamp())
        }
        
        # Add drifting features
        if drifting_features:
            attachment["fields"].append({
                "title": "Drifting Features",
                "value": "\n".join(f"• {feature}" for feature in drifting_features[:10]),
                "short": False
            })
        
        # Add summary
        attachment["fields"].append({
            "title": "Summary",
            "value": (
                f"Total Features: {drift_results['total_features']}\n"
                f"Drift Percentage: {drift_results['drift_percentage']:.1%}\n"
                f"Recommendation: {'Retrain model' if drift_results['overall_drift'] else 'Monitor'}"
            ),
            "short": False
        })
        
        return [attachment]
