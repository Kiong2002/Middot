"""Flask 扩展初始化"""

from flask import Flask
from openai import OpenAI
from .config import DEEPSEEK_API_KEY

llm_client: OpenAI | None = None


def init_extensions(app: Flask):
    """初始化所有 Flask 扩展"""
    global llm_client
    if DEEPSEEK_API_KEY:
        llm_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com",
        )
