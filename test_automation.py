
import os
import re
import json
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from google import genai

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# SETTINGS
# ============================================================

URL = "https://abhyas.ai"

SUBJECT = "ARITHMETIC"
CHAPTER = "PERCENTAGE"
QUESTION_COUNT = "30"
DIFFICULTY = "Average"
TIME_PER_QUESTION = "1 Mins"
PREVIOUS_PAPERS = "No"

# API model IDs. Verify that the IDs are valid in your provider consoles.
GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

if not GROQ_API_KEY:
    print("GROQ_API_KEY missing; Groq fallback disabled.")

# Initialize Gemini always. Groq is optional.
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY and Groq is not None else None


# ============================================================
# AI ANSWER HELPERS
# ============================================================

def parse_ai_answer(raw_text, options):
    if not raw_text:
        raise ValueError("The AI returned an empty response.")

    raw = raw_text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)

    result = json.loads(raw)
    choice = int(result["option"])
    explanation = str(result.get("reason", "")).strip()

    if choice < 1 or choice > len(options):
        raise ValueError("The AI returned an option number outside the available range.")

    return choice, explanation


def build_prompt(question_text, options):
    numbered_options = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(options))
    return f"""You are solving a multiple-choice arithmetic practice question.

Read the question carefully. Calculate the answer independently and compare it
against the available options. Return one valid JSON object only, in this format:
{{"option": 1, "reason": "Short calculation or explanation"}}

Rules:
- option must be a 1-based integer matching one of the listed options.
- Never invent an option that is not listed.
- If the question is ambiguous, choose the most defensible listed option and
  explain the ambiguity briefly.
- Do not include Markdown fences or text outside the JSON object.

QUESTION:
{question_text}

OPTIONS:
{numbered_options}
"""


def ask_gemini(prompt):
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    return response.text


def ask_groq(prompt):
    try:
        print("Sending request to Groq...")

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Solve the multiple-choice question carefully. "
                        "Return only valid JSON with keys option and reason. "
                        "The option must be a 1-based integer."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        print("Groq request succeeded.")
        return response.choices[0].message.content

    except Exception as exc:
        print("\n========== GROQ ERROR DETAILS ==========")
        print("Error type:", type(exc).__name__)
        print("HTTP status:", getattr(exc, "status_code", "Unknown"))

        # Extract the response body without printing API credentials.
        response = getattr(exc, "response", None)

        if response is not None:
            try:
                print("Response body:", response.text[:2000])
            except Exception:
                print("Response body could not be read.")

        # Some SDK errors expose the body through their attributes.
        body = getattr(exc, "body", None)

        if body is not None:
            print("Error body:", str(body)[:2000])

        print("========================================\n")

        raise


def ask_ai(question_text, options):
    prompt = build_prompt(question_text, options)

    providers = [("Gemini", ask_gemini)]

    if groq_client is not None:
        providers.append(("Groq", ask_groq))
    else:
        print("Groq fallback is not configured.")

    failures = []

    for provider_name, provider_call in providers:
        print(f"\nRequesting an answer from {provider_name}...")

        # Stage 1: Test the provider API call.
        try:
            raw_text = provider_call(prompt)
            print(f"{provider_name} API call succeeded.")

        except Exception as exc:
            print(f"\n========== {provider_name} API ERROR ==========")
            print("Error type:", type(exc).__name__)
            print("Error message:", str(exc)[:2000])

            # Print useful HTTP details when available.
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    print("HTTP status:", response.status_code)
                    print("Response body:", response.text[:1500])
                except Exception:
                    pass

            print("==============================================")

            failures.append(f"{provider_name} API: {type(exc).__name__}")
            continue

        # Stage 2: Test whether the returned answer is valid.
        try:
            answer = parse_ai_answer(raw_text, options)
            print(f"Answer validated successfully from {provider_name}.")
            return answer

        except Exception as exc:
            print(f"\n========== {provider_name} PARSE ERROR ==========")
            print("Error type:", type(exc).__name__)
            print("Error message:", str(exc)[:1000])
            print("Raw response:", str(raw_text)[:1500])
            print("================================================")

            failures.append(
                f"{provider_name} parsing: {type(exc).__name__}"
            )

    raise RuntimeError(
        "All AI providers failed. Details: " + "; ".join(failures)
    )

# ============================================================
# PAGE HELPERS
# ============================================================

def get_body_text(page):
    return page.locator("body").inner_text(timeout=5000)


def get_options(page):
    radios = page.locator("input[type='radio'][name='option']")
    radios.first.wait_for(state="attached", timeout=15000)

    labels = radios.evaluate_all("""
        elements => elements.map(el => {
            let label = null;

            if (el.id) {
                label = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
            }

            if (!label) label = el.closest('label');

            if (!label && el.nextElementSibling && el.nextElementSibling.tagName === 'LABEL') {
                label = el.nextElementSibling;
            }

            if (!label) {
                const parent = el.parentElement;
                if (parent && parent.querySelectorAll('input[type=radio]').length === 1) {
                    label = parent;
                }
            }

            return label ? label.innerText.trim() : (el.parentElement?.innerText || '').trim();
        })
    """)

    options = [label for label in labels if label]
    if len(options) != radios.count():
        raise RuntimeError("Could not reliably identify all answer options. Stopping safely.")

    return radios, options


def get_question_number(body):
    match = re.search(r"\b(\d+)\s*/\s*(\d+)\b", body)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def select_option(radios, index):
    radio = radios.nth(index)
    try:
        radio.check(timeout=4000, force=True)
    except Exception:
        radio.evaluate("el => el.click()")

    if not radio.is_checked():
        raise RuntimeError("The selected answer was not checked.")


# ============================================================
# MAIN AUTOMATION
# ============================================================

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=False)
    context = browser.new_context()
    page = context.new_page()

    try:
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        except PlaywrightTimeoutError:
            print("The page loaded slowly; checking whether it is usable...")

        input("\nLog in manually and reach the dashboard. Then press Enter here...")

        page.locator("a").filter(has_text=re.compile(r"^\s*Practice Tests\s*$")).first.click()
        page.get_by_text("CRT", exact=True).click()

        page.locator("div[role='combobox'] input").first.click()
        page.get_by_role("option", name=SUBJECT, exact=True).click()
        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()

        page.locator("div[role='combobox'] input").last.click()
        page.get_by_role("option", name=CHAPTER, exact=True).click()
        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()

        question_radio_ids = {"10": "cstm-radio1", "20": "cstm-radio2", "30": "cstm-radio3"}
        if QUESTION_COUNT not in question_radio_ids:
            raise ValueError("QUESTION_COUNT must be 10, 20, or 30.")
        page.locator(f"label[for='{question_radio_ids[QUESTION_COUNT]}']").click()

        difficulty_radio_ids = {"Easy": "cstm-level-radio1", "Average": "cstm-level-radio2", "Difficult": "cstm-level-radio3"}
        if DIFFICULTY not in difficulty_radio_ids:
            raise ValueError("DIFFICULTY must be Easy, Average, or Difficult.")
        page.locator(f"label[for='{difficulty_radio_ids[DIFFICULTY]}']").click()

        page.locator("select[formcontrolname='qtime']").select_option(label=TIME_PER_QUESTION)
        page.locator("select[formcontrolname='qtype']").select_option(label=" Single Answer ")
        page.locator("select[formcontrolname='pep']").select_option(label=PREVIOUS_PAPERS)

        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()
        print("\n--- TEST PREVIEW ---")
        print(get_body_text(page))
        input("\nReview the settings. Press Enter to continue...")

        start_confirmation = input("Type START to begin the practice test: ").strip().upper()
        if start_confirmation != "START":
            print("Start cancelled. Leaving the test setup open.")
        else:
            page.get_by_text("Start", exact=True).click()
            page.locator("input[type='radio'][name='option']").first.wait_for(state="visible", timeout=20000)

            print("\nTest opened. AI-assisted solving begins.")
            answered = 0

            while True:
                body = get_body_text(page)
                current, total = get_question_number(body)

                if current is None or total is None:
                    print("\nCould not identify the question number. Stopping.")
                    print(body)
                    break

                radios, options = get_options(page)

                print("\n" + "=" * 55)
                print(f"QUESTION {current}/{total}")
                print("=" * 55)
                print(body)
                print("\nDetected options:")
                for i, option in enumerate(options, start=1):
                    print(f"{i}. {option}")

                try:
                    choice, explanation = ask_ai(body, options)
                except Exception:
                    print("\nAI request failed. No answer selected; stopping safely.")
                    break

                print(f"\nAI suggests option {choice}:")
                print(options[choice - 1])
                print(f"Reason: {explanation}")

                try:
                    select_option(radios, choice - 1)
                except Exception as exc:
                    print(f"\nCould not select the answer ({type(exc).__name__}).")
                    print("Stopping without advancing.")
                    break

                answered += 1
                print("\nAnswer selected.")

                if current >= total:
                    print("\nAll questions have been processed.")
                    print("Review your answers in the browser before deciding whether to submit.")
                    confirmation = input("\nType SUBMIT to attempt submission, or REVIEW to leave the test open: ").strip().upper()

                    if confirmation == "SUBMIT":
                        submit_button = page.get_by_role("button", name=re.compile(r"submit", re.I))
                        visible_submit = [submit_button.nth(i) for i in range(submit_button.count()) if submit_button.nth(i).is_visible()]

                        if not visible_submit:
                            print("No visible Submit button found. Submit manually in the browser.")
                            input("Press Enter when you are finished...")
                            break

                        visible_submit[-1].click()
                        page.wait_for_timeout(700)
                        yes_button = page.get_by_role("button", name=re.compile(r"^Yes$", re.I))

                        if yes_button.count() > 0 and yes_button.last.is_visible():
                            final_confirmation = input("A confirmation dialog appeared. Type YES to confirm submission: ").strip().upper()
                            if final_confirmation == "YES":
                                yes_button.last.click()
                                print("Submission confirmation clicked. Verify the result in the browser.")
                            else:
                                print("Confirmation cancelled. Check the browser.")
                        else:
                            print("Submit was clicked. Check the browser to verify the result.")
                    else:
                        print("Test left open for you to review and submit manually.")
                    break

                previous_question = current
                page.locator("button.right-icon").click()

                try:
                    page.wait_for_function(
                        """previous => {
                            const text = document.body.innerText;
                            const match = text.match(/\\b(\\d+)\\s*\\/\\s*(\\d+)\\b/);
                            return match && Number(match[1]) !== previous;
                        }""",
                        arg=previous_question,
                        timeout=10000,
                    )
                except PlaywrightTimeoutError:
                    print("\nThe question number did not change. Stopping to avoid repeating an answer.")
                    break

            print(f"\nQuestions answered by automation: {answered}")

    except Exception as exc:
        print(f"\nAUTOMATION ERROR: {type(exc).__name__}")
        try:
            print(get_body_text(page))
        except Exception:
            pass

    finally:
        input("\nPress Enter to close Chrome...")
        browser.close()
