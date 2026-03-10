# week8-project/agents/messaging_agent.py

import os
import logging
import requests
from dotenv import load_dotenv
from openai import OpenAI          # already used in scraper_agent — no new dep
from typing import Any, Dict, List

load_dotenv(override=True)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


class MessagingAgent:
    name = "Messaging Agent"

    def __init__(self):
        self.pushover_user  = os.getenv("PUSHOVER_USER")
        self.pushover_token = os.getenv("PUSHOVER_TOKEN")
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )

    def _find_best_deal(self, products: List[Dict]) -> Dict | None:
        """Pick the product with the highest savings_pct that isn't Overpriced."""
        eligible = [p for p in products if p.get("verdict") != "Overpriced"]
        if not eligible:
            eligible = products          # fall back to all if none qualify
        return max(eligible, key=lambda p: float(p.get("savings_pct", 0)), default=None)

    def _craft_message(self, product: Dict) -> str:
        """Use LLM to write an exciting 2-sentence push notification."""
        messages = [
            {"role": "system", "content": """You are a helpful assistant that writes exciting push notifications for deals that have found by comparing pricess across various offers.
            You must be clear, concise and to the point as these appear as push notifications to the user
            """},
            {"role": "user", "content": f"Write a 2-sentence push notification for the following deal: {product}"}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content

    def push(self, message: str) -> None:
        """Send a raw push notification via Pushover."""
        payload = {
            "user": self.pushover_user,
            "token": self.pushover_token,
            "message": message,
            "sound": "cashregister",
        }
        requests.post(PUSHOVER_URL, data=payload)

    def notify_best_deal(self, products: List[Dict]) -> None:
        """Find the best deal and push a notification about it."""
        best = self._find_best_deal(products)
        if not best:
            return
        message = self._craft_message(best)
        self.push(message)