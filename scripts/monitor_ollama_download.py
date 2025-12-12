#!/usr/bin/env python3
"""
Ollama Download Monitor with Email Notification
Monitors Ollama model downloads on dgx-spark server and sends email when complete.
"""

import subprocess
import json
import time
import sys
import argparse
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict, Any


class OllamaMonitor:
    """Monitor Ollama model downloads and send notifications."""

    def __init__(
        self,
        model_name: str,
        server: str = "dgx-spark",
        check_interval: int = 30,
        email_to: Optional[str] = None,
        email_from: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
    ):
        """
        Initialize the monitor.

        Args:
            model_name: Name of the Ollama model to monitor (e.g., "deepseek-r1:32b")
            server: SSH server hostname (default: dgx-spark)
            check_interval: Seconds between checks (default: 30)
            email_to: Recipient email address
            email_from: Sender email address
            smtp_server: SMTP server address
            smtp_port: SMTP port (default: 587)
            smtp_user: SMTP username
            smtp_password: SMTP password
        """
        self.model_name = model_name
        self.server = server
        self.check_interval = check_interval
        self.email_to = email_to
        self.email_from = email_from or email_to
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password

        self.start_time = datetime.now()
        self.last_status = None

    def check_model_status(self) -> Dict[str, Any]:
        """
        Check if the model exists in Ollama on the remote server.

        Returns:
            Dictionary with status information:
                - exists: bool - whether model exists
                - size: int - model size in bytes (if exists)
                - error: str - error message (if any)
        """
        try:
            # List all models via Ollama API
            cmd = f'ssh {self.server} "curl -s http://localhost:11434/api/tags"'
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return {
                    "exists": False,
                    "error": f"SSH/API error: {result.stderr}"
                }

            # Parse JSON response
            try:
                data = json.loads(result.stdout)
                models = data.get("models", [])

                # Check if our model exists
                for model in models:
                    if model.get("name") == self.model_name:
                        return {
                            "exists": True,
                            "size": model.get("size", 0),
                            "modified": model.get("modified_at", ""),
                        }

                return {"exists": False}

            except json.JSONDecodeError as e:
                return {
                    "exists": False,
                    "error": f"JSON parse error: {e}"
                }

        except subprocess.TimeoutExpired:
            return {
                "exists": False,
                "error": "Connection timeout"
            }
        except Exception as e:
            return {
                "exists": False,
                "error": f"Unexpected error: {e}"
            }

    def send_email(self, subject: str, body: str) -> bool:
        """
        Send email notification.

        Args:
            subject: Email subject
            body: Email body (plain text)

        Returns:
            True if successful, False otherwise
        """
        if not all([self.email_to, self.smtp_server]):
            print("⚠️  Email not configured, skipping notification")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = subject

            msg.attach(MIMEText(body, 'plain'))

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            print(f"✅ Email sent to {self.email_to}")
            return True

        except Exception as e:
            print(f"❌ Email error: {e}")
            return False

    def format_size(self, bytes: int) -> str:
        """Format bytes to human-readable size."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"

    def format_duration(self, seconds: int) -> str:
        """Format seconds to human-readable duration."""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0:
            parts.append(f"{int(minutes)}m")
        if seconds > 0 or not parts:
            parts.append(f"{int(seconds)}s")

        return " ".join(parts)

    def monitor(self, max_checks: Optional[int] = None) -> None:
        """
        Monitor the download until completion.

        Args:
            max_checks: Maximum number of checks (None = unlimited)
        """
        print(f"🔍 Starting monitor for model: {self.model_name}")
        print(f"📡 Server: {self.server}")
        print(f"⏱️  Check interval: {self.check_interval}s")
        if self.email_to:
            print(f"📧 Email notifications: {self.email_to}")
        print("-" * 60)

        check_count = 0

        while True:
            check_count += 1

            if max_checks and check_count > max_checks:
                print(f"\n⏹️  Reached maximum checks ({max_checks})")
                break

            # Check status
            status = self.check_model_status()

            if status.get("error"):
                print(f"\n⚠️  Error: {status['error']}")

            elif status.get("exists"):
                # Model download complete!
                elapsed = (datetime.now() - self.start_time).total_seconds()
                size_str = self.format_size(status.get("size", 0))
                duration_str = self.format_duration(int(elapsed))

                print(f"\n✅ Download complete!")
                print(f"📦 Model: {self.model_name}")
                print(f"💾 Size: {size_str}")
                print(f"⏱️  Duration: {duration_str}")

                # Send email notification
                if self.email_to:
                    subject = f"✅ Ollama Download Complete: {self.model_name}"
                    body = f"""
Ollama model download completed successfully!

Model: {self.model_name}
Server: {self.server}
Size: {size_str}
Duration: {duration_str}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The model is now available for use.
"""
                    self.send_email(subject, body)

                break

            else:
                # Still downloading
                elapsed = (datetime.now() - self.start_time).total_seconds()
                duration_str = self.format_duration(int(elapsed))

                print(f"⏳ Check #{check_count}: Model not ready yet (elapsed: {duration_str})", end="\r")

            # Wait before next check
            time.sleep(self.check_interval)

        print("\n" + "=" * 60)
        print("🏁 Monitoring complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor Ollama model downloads with email notifications"
    )

    parser.add_argument(
        "model",
        help="Model name to monitor (e.g., deepseek-r1:32b)"
    )

    parser.add_argument(
        "--server",
        default="dgx-spark",
        help="SSH server hostname (default: dgx-spark)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )

    parser.add_argument(
        "--email-to",
        help="Recipient email address"
    )

    parser.add_argument(
        "--email-from",
        help="Sender email address (defaults to --email-to)"
    )

    parser.add_argument(
        "--smtp-server",
        help="SMTP server address (e.g., smtp.gmail.com)"
    )

    parser.add_argument(
        "--smtp-port",
        type=int,
        default=587,
        help="SMTP port (default: 587)"
    )

    parser.add_argument(
        "--smtp-user",
        help="SMTP username"
    )

    parser.add_argument(
        "--smtp-password",
        help="SMTP password"
    )

    parser.add_argument(
        "--max-checks",
        type=int,
        help="Maximum number of checks (default: unlimited)"
    )

    args = parser.parse_args()

    # Create monitor
    monitor = OllamaMonitor(
        model_name=args.model,
        server=args.server,
        check_interval=args.interval,
        email_to=args.email_to,
        email_from=args.email_from,
        smtp_server=args.smtp_server,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password=args.smtp_password,
    )

    try:
        monitor.monitor(max_checks=args.max_checks)
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
