
import os
import json
import re
import time
from pathlib import Path
from datetime import datetime

import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_URL = "https://abhyas.ai"
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OUTPUT_DIR = Path("exam_debug")
OUTPUT_DIR.mkdir(exist_ok=True)

TIMEOUT_MS = 15000

# ============================================================
# GROQ CONNECTION TEST
# ============================================================

def test_groq_connection():
    """Verify the configured Groq model before opening the test."""

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: CONNECTION_OK",
            }
        ],
        "temperature": 0,
        "max_tokens": 30,
    }

    response = requests.post(
        GROQ_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            f"Groq HTTP {response.status_code}: "
            f"{response.text[:1500]}"
        )

    result = response.json()
    answer = result["choices"][0]["message"]["content"]

    print(f"[Groq] Model: {GROQ_MODEL}")
    print(f"[Groq] Connection successful: {answer.strip()}")


# ============================================================
# DEBUGGING AND PAGE INSPECTION
# ============================================================

def save_debug_artifacts(page, label="page"):
    """Save a screenshot and DOM snapshot for troubleshooting."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = OUTPUT_DIR / f"{label}_{timestamp}"

    try:
        page.screenshot(
            path=str(prefix.with_suffix(".png")),
            full_page=True,
        )
        print(f"[Debug] Screenshot: {prefix}.png")
    except Exception as exc:
        print(f"[Debug] Screenshot failed: {exc}")

    try:
        prefix.with_suffix(".html").write_text(
            page.content(),
            encoding="utf-8",
        )
        print(f"[Debug] DOM saved: {prefix}.html")
    except Exception as exc:
        print(f"[Debug] DOM capture failed: {exc}")


def visible_text(page):
    try:
        return page.locator("body").inner_text(timeout=TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return ""


def inspect_controls(page):
    """Print visible controls and their accessible labels."""

    print("\n--- VISIBLE CONTROLS ---")

    for selector in [
        "button",
        '[role="button"]',
        'input[type="radio"]',
        'input[type="checkbox"]',
        '[role="radio"]',
        '[role="checkbox"]',
        "textarea",
        "select",
    ]:
        locator = page.locator(selector)

        try:
            count = locator.count()

            for index in range(min(count, 40)):
                element = locator.nth(index)

                try:
                    if not element.is_visible():
                        continue

                    print({
                        "selector": selector,
                        "index": index,
                        "text": element.inner_text(timeout=1000)[:250],
                        "aria_label": element.get_attribute("aria-label"),
                        "type": element.get_attribute("type"),
                        "disabled": not element.is_enabled(),
                    })

                except Exception:
                    continue

        except Exception as exc:
            print(f"[Inspect] {selector}: {exc}")


# ============================================================
# DASHBOARD NAVIGATION
# ============================================================

def find_continue_test(page):
    """Find the single CONTINUE control."""

    candidates = page.get_by_role(
        "button",
        name="START TEST",
        exact=True,
    )

    if candidates.count() == 1:
        return candidates

    # Some sites implement clickable cards using non-button elements.
    candidates = page.get_by_text(
        "START TEST",
        exact=True,
    )

    if candidates.count() == 1:
        return candidates

    raise RuntimeError(
        "START TEST could not be uniquely identified. "
        "No click was attempted."
    )


def open_test(page):
    print("\n[Navigation] Opening Abhyas.ai...")

    page.goto(
        BASE_URL,
        wait_until="domcontentloaded",
        timeout=60000,
    )

    input(
        "\nLog in manually and navigate to the intended test card. "
        "Press Enter when the dashboard is ready..."
    )

    continue_button = find_continue_test(page)

    print("[Navigation] Clicking START TEST...")
    continue_button.click()

    page.wait_for_load_state(
        "domcontentloaded",
        timeout=30000,
    )

    page.wait_for_timeout(1500)

    print("[Navigation] Current URL:", page.url)
    print("[Navigation] Page title:", page.title())


# ============================================================
# QUESTION CAPTURE
# ============================================================

def capture_current_page(page):
    """
    Capture visible page text and controls.

    This deliberately does not guess which option is correct
    or select answers automatically.
    """

    text = visible_text(page)

    print("\n" + "=" * 65)
    print("CURRENT PAGE")
    print("=" * 65)
    print("URL:", page.url)
    print(text[:10000])

    inspect_controls(page)

    return text


# ============================================================
# MANUAL QUESTION NAVIGATION
# ============================================================

def find_next_control(page):
    """Find an unambiguous Next control."""

    patterns = [
        re.compile(r"^next$", re.IGNORECASE),
        re.compile(r"^next question$", re.IGNORECASE),
    ]

    matches = []

    for pattern in patterns:
        for role in ("button", "link"):
            locator = page.get_by_role(role, name=pattern)

            try:
                for index in range(locator.count()):
                    item = locator.nth(index)

                    if item.is_visible() and item.is_enabled():
                        matches.append(item)
            except Exception:
                continue

    # Deduplicate matches by DOM identity.
    unique = []

    for item in matches:
        try:
            if not any(item.evaluate(
                "(el) => el === other",
                arg=other.evaluate("(el) => el")
            ) for other in []):
                unique.append(item)
        except Exception:
            unique.append(item)

    if len(unique) == 1:
        return unique[0]

    return None


def run_question_viewer(page):
    """
    Display each question for manual review.
    The user controls answers and all exam submissions.
    """

    while True:
        text = capture_current_page(page)

        if not text.strip():
            print("[Warning] No visible page text was captured.")

        if re.search(
            r"\b(submit test|finish test|test completed|"
            r"view result|exam submitted)\b",
            text,
            re.IGNORECASE,
        ):
            print(
                "\nA possible final-test screen was detected. "
                "Review it yourself before taking any action."
            )
            break

        command = input(
            "\nEnter 'n' to go to the next question, "
            "'s' to save a screenshot, or 'q' to stop: "
        ).strip().lower()

        if command == "q":
            print("[Navigation] Stopped by user.")
            break

        if command == "s":
            save_debug_artifacts(page, "question")
            continue

        if command != "n":
            print("Unknown command.")
            continue

        next_button = find_next_control(page)

        if next_button is None:
            print(
                "[Navigation] No unique enabled Next control found. "
                "No click was attempted."
            )
            save_debug_artifacts(page, "next_not_found")
            break

        try:
            next_button.click()
            page.wait_for_timeout(700)
        except Exception as exc:
            print("[Navigation] Could not advance:", exc)
            save_debug_artifacts(page, "navigation_error")
            break


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("ABHYAS.AI BROWSER ASSISTANT")
    print("=" * 65)

    test_groq_connection()

    if not Path(CHROME_PATH).exists():
        raise FileNotFoundError(
            f"Chrome executable not found: {CHROME_PATH}"
        )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=CHROME_PATH,
            headless=False,
        )

        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(TIMEOUT_MS)

        try:
            open_test(page)

            print("\n[Test] The test page is open.")
            print(
                "Review each question and choose answers in the "
                "browser yourself."
            )

            run_question_viewer(page)

        except KeyboardInterrupt:
            print("\nStopped by keyboard interrupt.")

        except Exception as exc:
            print("\nAUTOMATION ERROR:", repr(exc))

            try:
                save_debug_artifacts(page, "automation_error")
            except Exception:
                pass

        finally:
            input(
                "\nPress Enter to close the browser "
                "(or use Ctrl+C to exit)..."
            )

            context.close()
            browser.close()


if __name__ == "__main__":
    main()
