"""
Telegram notifications for the screener.

Needs two values, set as environment variables (in GitHub Actions: add them
as repository secrets, see SETUP below for exactly how):

  TELEGRAM_BOT_TOKEN  - identifies your bot
  TELEGRAM_CHAT_ID    - identifies where to send the message (you, or a group)

SETUP (one-time, ~2 minutes)
---------------------------------------------------------------
1. In Telegram, search for "@BotFather", start a chat with it.
2. Send: /newbot
   Follow the prompts (pick a display name, then a username that must end
   in "bot", e.g. "anchu_screener_bot"). BotFather replies with a token
   that looks like  123456789:AAEhBOweik9ai3...  - that's TELEGRAM_BOT_TOKEN.
3. Send YOUR new bot any message (e.g. "hi") - bots can't message you first.
4. Get your chat ID - open this URL in a browser, with your real token:
       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   Find "chat":{"id": 123456789, ...} in the response. That number is
   TELEGRAM_CHAT_ID.
5. In your GitHub repo: Settings -> Secrets and variables -> Actions ->
   New repository secret. Add both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

import os

import requests

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "TELEGRAM_CHAT_ID"


def send_telegram_message(text, parse_mode="Markdown"):
    """Sends a message. If the env vars aren't set, prints a warning and
    returns False instead of crashing - a missing Telegram config should
    never take down the actual screener run."""
    token = os.environ.get(BOT_TOKEN_ENV)
    chat_id = os.environ.get(CHAT_ID_ENV)
    if not token or not chat_id:
        print(f"Telegram not configured ({BOT_TOKEN_ENV}/{CHAT_ID_ENV} missing) - skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = text[:4000]  # Telegram's hard cap is 4096 chars; trim defensively
    try:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Telegram send failed: {e}")
        return False
