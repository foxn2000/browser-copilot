from groq import Groq
import google.generativeai as genai

import os
import platform
import sys
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent, SystemPrompt, Browser, BrowserConfig
import asyncio
from functools import wraps

from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(verbose=True)

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)

if platform.system() == "Windows":
    chrome_path = os.environ.get("CHROME_BROWSER_PATH_WINDOWS")
elif platform.system() == "Darwin":  # macOS の場合
    chrome_path = os.environ.get("CHROME_BROWSER_PATH_MAC")
else:
    chrome_path = os.environ.get("CHROME_BROWSER_PATH")  # その他の環境向け

client_groq = Groq(api_key=os.environ.get("GROQ_API_KEY"))
genai.configure(api_key = os.environ.get("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
あなたは優秀なAIアシスタントです。
"""

def gemini_chat(user_inputs,system_prompt = SYSTEM_PROMPT,main_model="gemini-2.0-flash"):
    gemini_model = genai.GenerativeModel(
        model_name= main_model,
        system_instruction=system_prompt,
        )
    massage_history = []
    for item in user_inputs[:-1]:
        if item["role"] == "user":
            massage_history.append({"role": "user", "parts": item["content"]})
        elif item["role"] == "assistant":
            massage_history.append({"role": "model", "parts": item["content"]})
    gemini_chat = gemini_model.start_chat(history = massage_history)
    response = gemini_chat.send_message(user_inputs[-1]["content"])
    return response.text

def extract_tag(text, output_type="text", tag_name="output"):
    if tag_name == "output":
        pattern = r"<output>(.*?)</output>"
    elif tag_name == "code":
        pattern = r"<code>(.*?)</code>"
    elif tag_name == "input":
        pattern = r"<input>(.*?)</input>"
    else:
        try:
            pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
        except:
            raise ValueError(
                "Invalid tag_name. Please choose from 'output', 'code' or 'input'."
            )
    matches = re.findall(
        pattern, text, re.DOTALL
    )  # re.DOTALL allows '.' to match newline characters
    if output_type == "text":
        return matches[0]
    elif output_type == "list":
        return matches


def sync_wrapper(async_func):
    @wraps(async_func)
    def sync_func(*args, **kwargs):
        return asyncio.run(async_func(*args, **kwargs))
    return sync_func

@sync_wrapper
async def gemini_agent(task, model_name:str="gemini-2.0-pro-exp-02-05"):
    # データベースから設定を取得
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("APIキーが設定されていません。設定画面から設定してください。")

    try:
        browser = Browser(
            config=BrowserConfig(
                chrome_instance_path=chrome_path,
            )
        )

        agent = Agent(
            task=task,
            llm=ChatGoogleGenerativeAI(
                model=model_name,
                api_key=api_key,
            ),
            browser=browser
        )
        result = await agent.run()
        return result.final_result()
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"
