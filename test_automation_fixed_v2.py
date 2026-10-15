import os
import re
import json
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from google import genai

try:
    from groq import Groq
except ImportError:
    Groq = None


# ============================================================
# SETTINGS — change these values if needed
# ============================================================

URL = "https://abhyas.ai"

SUBJECT = "ARITHMETIC"
CHAPTER = "PERCENTAGE"
QUESTION_COUNT = "30"       # Supported values: "10", "20", "30"
DIFFICULTY = "Average"       # "Easy", "Average", "Difficult"
TIME_PER_QUESTION = "2 Mins"
PREVIOUS_PAPERS = "No"      # "Yes" or "No"

# Model IDs can change. Use IDs available to your API accounts.
GEMINI_MODEL = "gemini-3.8-flash"

GROQ_MODEL = "openai/gpt-oss-120b"



# ============================================================
# API CLIENT SETUP
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

groq_client = None
if GROQ_API_KEY and Groq is not None:
    groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# AI ANSWER HELPERS — Gemini first, Groq fallback
# ============================================================

def parse_ai_answer(raw_text, options):
    """Parse and validate a provider's JSON response."""
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
    numbered_options = "\n".join(
        f"{i + 1}. {option}" for i, option in enumerate(options)
    )

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
    if groq_client is None:
        raise RuntimeError("Groq is not configured with an API key.")

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Solve multiple-choice arithmetic carefully. Return valid JSON only "
                    'with keys "option" (1-based integer) and "reason" (string).'
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def ask_ai(question_text, options):
    """Try Gemini first; if it fails, try Groq. Never silently guess."""
    prompt = build_prompt(question_text, options)
    providers = [("Gemini", ask_gemini)]

    if groq_client is not None:
        providers.append(("Groq", ask_groq))
    else:
        print("Groq fallback is not configured; using Gemini only.")

    failures = []
    for provider_name, provider_call in providers:
        try:
            print(f"Requesting an answer from {provider_name}...")
            raw_text = provider_call(prompt)
            answer = parse_ai_answer(raw_text, options)
            print(f"Answer received and validated from {provider_name}.")
            return answer
        except Exception as exc:
            # Do not print raw provider exception text; keep API details private.
            print(f"{provider_name} failed ({type(exc).__name__}).")
            failures.append(provider_name)

    raise RuntimeError(
        "No AI provider returned a valid answer. Tried: " + ", ".join(failures)
    )


# ============================================================
# PAGE HELPERS
# ============================================================

def normalize_option_label(value):
    """Normalize dropdown labels such as '2 Mins' and '2 Min'."""
    value = " ".join((value or "").strip().lower().split())
    value = re.sub(r"\bminutes\b", "min", value)
    value = re.sub(r"\bmins\b", "min", value)
    return value


def select_dropdown_option(page, selector, desired_label, field_name):
    """Select a native dropdown option by normalized visible label."""
    dropdown = page.locator(selector)
    dropdown.wait_for(state="visible", timeout=15000)

    option_rows = dropdown.locator("option").evaluate_all(
        "els => els.map(el => ({label: (el.textContent || '').trim(), value: el.value}))"
    )
    print(f"[DEBUG] {field_name} dropdown options: {option_rows}")

    desired = normalize_option_label(desired_label)
    match = next(
        (row for row in option_rows if normalize_option_label(row["label"]) == desired),
        None,
    )

    if match is None:
        raise RuntimeError(
            f"Could not match {field_name} option {desired_label!r}. "
            f"Available options: {[row['label'] for row in option_rows]}"
        )

    # Select by value where possible; fall back to the exact observed label.
    if match["value"] != "":
        dropdown.select_option(value=match["value"], timeout=15000)
    else:
        dropdown.select_option(label=match["label"], timeout=15000)

    selected_label = dropdown.locator("option:checked").inner_text().strip()
    print(f"[STEP] {field_name} selected: {selected_label}")


def get_body_text(page):
    return page.locator("body").inner_text(timeout=5000)


def get_options(page):
    radios = page.locator("input[type='radio'][name='option']")
    radios.first.wait_for(state="attached", timeout=15000)

    labels = radios.evaluate_all("""
        elements => elements.map(el => {
            let label = null;

            if (el.id) {
                label = document.querySelector(
                    'label[for="' + CSS.escape(el.id) + '"]'
                );
            }

            if (!label) label = el.closest("label");

            if (!label && el.nextElementSibling &&
                el.nextElementSibling.tagName === "LABEL") {
                label = el.nextElementSibling;
            }

            if (!label) {
                const parent = el.parentElement;
                if (parent && parent.querySelectorAll("input[type=radio]").length === 1) {
                    label = parent;
                }
            }

            return label
                ? label.innerText.trim()
                : (el.parentElement?.innerText || "").trim();
        })
    """)

    options = [label for label in labels if label]
    if len(options) != radios.count():
        raise RuntimeError(
            "Could not reliably identify all answer options. Stopping safely."
        )

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
        # 1. Login manually
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        except PlaywrightTimeoutError:
            print("The page loaded slowly; checking whether it is usable...")

        input("\nLog in manually and reach the dashboard. Then press Enter here...")

        # 2. Open practice test creation
        page.locator("a").filter(
            has_text=re.compile(r"^\s*Practice Tests\s*$")
        ).first.click()

        page.get_by_text("CRT", exact=True).click()

        # Subject
        page.locator("div[role='combobox'] input").first.click()
        page.get_by_role("option", name=SUBJECT, exact=True).click()
        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()

        # Chapter
        page.locator("div[role='combobox'] input").last.click()
        page.get_by_role("option", name=CHAPTER, exact=True).click()
        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()

        # Number of questions
        question_radio_ids = {
            "10": "cstm-radio1",
            "20": "cstm-radio2",
            "30": "cstm-radio3",
        }
        if QUESTION_COUNT not in question_radio_ids:
            raise ValueError("QUESTION_COUNT must be 10, 20, or 30.")
        page.locator(
            f"label[for='{question_radio_ids[QUESTION_COUNT]}']"
        ).click()

        # Difficulty
        difficulty_radio_ids = {
            "Easy": "cstm-level-radio1",
            "Average": "cstm-level-radio2",
            "Difficult": "cstm-level-radio3",
        }
        if DIFFICULTY not in difficulty_radio_ids:
            raise ValueError("DIFFICULTY must be Easy, Average, or Difficult.")
        page.locator(
            f"label[for='{difficulty_radio_ids[DIFFICULTY]}']"
        ).click()

        # Time per question, question type, and previous papers
        select_dropdown_option(
            page,
            "select[formcontrolname='qtime']",
            TIME_PER_QUESTION,
            "Time per question",
        )
        select_dropdown_option(
            page,
            "select[formcontrolname='qtype']",
            "Single Answer",
            "Question type",
        )
        select_dropdown_option(
            page,
            "select[formcontrolname='pep']",
            PREVIOUS_PAPERS,
            "Previous papers",
        )

        # 3. Review settings before starting
        page.get_by_role("button", name=re.compile(r"^Next$", re.I)).click()
        print("\n--- TEST PREVIEW ---")
        print(get_body_text(page))
        input("\nReview the settings. Press Enter to continue...")

        start_confirmation = input("Type START to begin the practice test: ").strip().upper()
        if start_confirmation != "START":
            print("Start cancelled. Leaving the test setup open.")
        else:
            page.get_by_text("Start", exact=True).click()
            page.locator(
                "input[type='radio'][name='option']"
            ).first.wait_for(state="visible", timeout=20000)

            print("\nTest opened. AI-assisted solving begins.")
            answered = 0

            # 4. Solve questions one at a time
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

                # Never automatically advance past the final question or submit.
                if current >= total:
                    print("\nAll questions have been processed.")
                    print("Review your answers in the browser before deciding whether to submit.")
                    confirmation = input(
                        "\nType SUBMIT to attempt submission, or REVIEW to leave the test open: "
                    ).strip().upper()

                    if confirmation == "SUBMIT":
                        submit_button = page.get_by_role(
                            "button", name=re.compile(r"submit", re.I)
                        )
                        visible_submit = [
                            submit_button.nth(i)
                            for i in range(submit_button.count())
                            if submit_button.nth(i).is_visible()
                        ]

                        if not visible_submit:
                            print("No visible Submit button found. Submit manually in the browser.")
                            input("Press Enter when you are finished...")
                            break

                        visible_submit[-1].click()
                        page.wait_for_timeout(700)
                        yes_button = page.get_by_role(
                            "button", name=re.compile(r"^Yes$", re.I)
                        )

                        if yes_button.count() > 0 and yes_button.last.is_visible():
                            final_confirmation = input(
                                "A confirmation dialog appeared. Type YES to confirm submission: "
                            ).strip().upper()
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

                # Advance only after an answer was selected.
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
        # Do not print full page URLs, which can sometimes contain session data.
        print(f"\nAUTOMATION ERROR: {type(exc).__name__}")
        try:
            print(get_body_text(page))
        except Exception:
            pass

    finally:
        input("\nPress Enter to close Chrome...")
        browser.close()
