import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_reset_email(to_email: str, reset_token: str) -> None:
    """Sends a password reset email using Gmail SMTP (same pattern as HamroDeal)."""
    gmail_user = os.getenv("EMAIL_USER")
    gmail_pass = os.getenv("EMAIL_PASS")
    client_url = os.getenv("CLIENT_URL", "http://localhost:3000")

    if not gmail_user or not gmail_pass:
        raise RuntimeError(
            "EMAIL_USER and EMAIL_PASS must be set in .env to send password reset emails."
        )

    reset_link = f"{client_url}/reset-password?token={reset_token}"

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 480px; margin: 0 auto; background: #111; color: #fff; padding: 40px; border-radius: 12px;">
      <h2 style="color: #00FF88; margin-bottom: 8px;">GymForm</h2>
      <h3 style="color: #fff; font-weight: 800; font-size: 22px;">Reset your password</h3>
      <p style="color: #888; line-height: 1.6;">
        We received a request to reset your password. Click the button below — the link expires in 1 hour.
      </p>
      <a href="{reset_link}"
         style="display: inline-block; margin: 24px 0; padding: 14px 32px; background: #fff; color: #000;
                font-weight: 800; font-size: 15px; border-radius: 10px; text-decoration: none; letter-spacing: 1px;">
        RESET PASSWORD
      </a>
      <p style="color: #555; font-size: 12px;">
        If you didn't request this, ignore this email — your account is safe.
      </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Reset your GymForm password"
    msg["From"] = gmail_user
    msg["To"] = to_email
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(gmail_user, gmail_pass)
        smtp.sendmail(gmail_user, to_email, msg.as_string())
