
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from your existing .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERROR: GROQ_API_KEY was not found in your .env file.")
    raise SystemExit(1)

client = Groq(api_key=api_key)

try:
    print("Fetching models available to your Groq API key...\n")

    models = client.models.list()

    available_models = sorted(
        model.id for model in models.data
    )

    if not available_models:
        print("No models were returned.")
    else:
        print("Models returned by Groq:\n")
        for model_id in available_models:
            print(model_id)

        print(f"\nTotal models: {len(available_models)}")

except Exception as exc:
    print("\nCould not retrieve Groq models.")
    print("Error type:", type(exc).__name__)

    # Do not print the API key or request headers.
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            print("HTTP status:", response.status_code)
            print("Response:", response.text[:1000])
        except Exception:
            print("Check your Groq API configuration and network connection.")
    else:
        print("Message:", str(exc)[:500])
