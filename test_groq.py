
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise SystemExit("GROQ_API_KEY is missing from .env")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": (
                    'Answer with JSON only: '
                    '{"answer":"B","explanation":"Test successful"}'
                )
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    print("Groq API test successful!")
    print(response.choices[0].message.content)

except Exception as exc:
    print("Groq API test failed.")
    print("Error type:", type(exc).__name__)
    print("Error:", str(exc)[:1000])
