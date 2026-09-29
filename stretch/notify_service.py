"""
stretch/notify_service.py — Free Push Notifications via ntfy.sh
==============================================================
Sends instant push notifications for meeting confirmations and reminders.

100% Free:
- Uses ntfy.sh (Open-source, public push notification service)
- Zero API keys required
- Zero subscription cost (Replaces paid Twilio / WhatsApp)

Usage:
1. Set NTFY_TOPIC in .env (e.g., NTFY_TOPIC=ai-scheduler-mysecret123)
2. Subscribe to that topic on your phone:
   - Install 'ntfy' app from Google Play / iOS App Store (or open https://ntfy.sh/your-topic in browser)
3. Instant push alerts pop up whenever a meeting is booked!
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

NTFY_DEFAULT_TOPIC = os.getenv("NTFY_TOPIC", "ai-meeting-scheduler-alerts")


def send_notification(
    title: str,
    message: str,
    topic: str = None,
    click_url: str = None,
    priority: str = "default",
    tags: list[str] = None,
) -> bool:
    """
    Publish a push notification via ntfy.sh.

    Args:
        title: Notification header
        message: Body text
        topic: ntfy topic name (overrides env)
        click_url: Optional URL to open when notification is tapped
        priority: min, low, default, high, urgent
        tags: Emoji tags (e.g. ['calendar', 'white_check_mark'])

    Returns:
        True if sent successfully, False otherwise
    """
    target_topic = topic or NTFY_DEFAULT_TOPIC
    url = f"https://ntfy.sh/{target_topic}"

    headers = {
        "Title": title,
        "Priority": priority,
    }

    if click_url:
        headers["Click"] = click_url

    if tags:
        headers["Tags"] = ",".join(tags)

    try:
        response = requests.post(
            url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=5,
        )
        if response.status_code == 200:
            logger.info(f"ntfy.sh notification delivered to topic: {target_topic}")
            return True
        else:
            logger.warning(f"ntfy.sh error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logger.warning(f"Failed to deliver push notification: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing ntfy.sh push notification...")
    topic = os.getenv("NTFY_TOPIC", "ai-meeting-scheduler-test-demo")
    print(f"Subscribe at: https://ntfy.sh/{topic}")
    success = send_notification(
        title="🗓️ Test Meeting Booked!",
        message="Your 1:1 with Rahul has been confirmed for 3:00 PM.",
        topic=topic,
        tags=["calendar", "tada"],
    )
    print(f"Delivered: {success}")
