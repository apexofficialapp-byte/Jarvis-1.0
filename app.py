"""
app.py
ChatGPT Web Automation Gateway

Install:
    pip install flask playwright gunicorn

Install Chromium:
    playwright install chromium

Recommended Render start command:
    gunicorn --workers 1 --threads 4 --timeout 180 app:app

Why one worker?
The persistent Playwright browser manager is process-local. Multiple Gunicorn
workers would create multiple independent Chromium sessions and could contend
for the same profile directory. Use one worker unless you intentionally design
separate browser profiles/managers.

Environment variables:
    PORT=10000
    HEADLESS=true
    CHATGPT_URL=https://chatgpt.com/
    PLAYWRIGHT_USER_DATA_DIR=./playwright_data
    CHAT_TIMEOUT_SECONDS=120
    BROWSER_LOCK_TIMEOUT_SECONDS=180
    MAX_PROMPT_LENGTH=10000
    LOG_LEVEL=INFO

Authentication:
    This application never accepts ChatGPT credentials and never reads,
    prints, uploads, or exposes authentication cookies/tokens.

    For first-time local authentication:
      1. Set HEADLESS=false.
      2. Start this application.
      3. The persistent Chromium profile will open.
      4. Manually authenticate at ChatGPT.
      5. Keep the profile directory.
      6. Restart with HEADLESS=true if desired.

Render note:
    Render's normal filesystem is ephemeral. Therefore a persistent login
    profile may disappear after a restart/redeploy unless persistent storage
    is configured and mounted at PLAYWRIGHT_USER_DATA_DIR.

Security note:
    This program intentionally does NOT implement CAPTCHA solving, stealth
    plugins, anti-bot bypassing, credential automation, rate-limit evasion,
    CDP exposure, cookie upload, or security-control circumvention.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from flask import Flask, jsonify, render_template_string, request
from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


def env_int(name: str, default: int, minimum: Optional[int] = None) -> int:
    value = os.environ.get(name)

    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except ValueError:
            result = default

    if minimum is not None:
        result = max(result, minimum)

    return result


@dataclass(frozen=True)
class Config:
    port: int
    headless: bool
    chatgpt_url: str
    user_data_dir: str
    chat_timeout_seconds: int
    browser_lock_timeout_seconds: int
    max_prompt_length: int
    log_level: str


def get_config() -> Config:
    return Config(
        port=env_int("PORT", 10000, 1),
        headless=env_bool("HEADLESS", True),
        chatgpt_url=os.environ.get(
            "CHATGPT_URL",
            "https://chatgpt.com/",
        ).strip(),
        user_data_dir=os.environ.get(
            "PLAYWRIGHT_USER_DATA_DIR",
            "./playwright_data",
        ).strip(),
        chat_timeout_seconds=env_int(
            "CHAT_TIMEOUT_SECONDS",
            120,
            10,
        ),
        browser_lock_timeout_seconds=env_int(
            "BROWSER_LOCK_TIMEOUT_SECONDS",
            180,
            1,
        ),
        max_prompt_length=env_int(
            "MAX_PROMPT_LENGTH",
            10000,
            1,
        ),
        log_level=os.environ.get(
            "LOG_LEVEL",
            "INFO",
        ).upper(),
    )


CONFIG = get_config()


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("chatgpt-web-automation")


# ============================================================================
# FLASK
# ============================================================================

app = Flask(__name__)

app.config.update(
    MAX_CONTENT_LENGTH=max(
        CONFIG.max_prompt_length + 4096,
        16384,
    ),
    JSON_SORT_KEYS=False,
)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class GatewayError(Exception):
    """Base application exception."""


class InvalidPromptError(GatewayError):
    """The submitted prompt is invalid."""


class AuthenticationRequiredError(GatewayError):
    """The persistent ChatGPT browser profile is not authenticated."""


class BrowserUnavailableError(GatewayError):
    """The browser could not be started or recovered."""


class PromptInputNotFoundError(GatewayError):
    """ChatGPT's prompt input could not be located."""


class SendButtonNotFoundError(GatewayError):
    """ChatGPT's send control could not be located."""


class ResponseTimeoutError(GatewayError):
    """ChatGPT did not finish producing a response within the timeout."""


class EmptyResponseError(GatewayError):
    """ChatGPT returned no usable visible response text."""


class BrowserBusyError(GatewayError):
    """Another request currently owns the browser."""


# ============================================================================
# SECURITY HEADERS
# ============================================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    # This page intentionally loads Tailwind from the official CDN requested
    # by the application specification.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.tailwindcss.com 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:;"
    )

    return response


# ============================================================================
# PROMPT VALIDATION
# ============================================================================

def validate_prompt(payload: Any) -> str:
    if payload is None:
        raise InvalidPromptError("Request body must contain JSON.")

    if not isinstance(payload, dict):
        raise InvalidPromptError("JSON body must be an object.")

    if "prompt" not in payload:
        raise InvalidPromptError("The 'prompt' field is required.")

    prompt = payload["prompt"]

    if not isinstance(prompt, str):
        raise InvalidPromptError("The 'prompt' field must be a string.")

    if not prompt.strip():
        raise InvalidPromptError("The prompt cannot be empty.")

    if len(prompt) > CONFIG.max_prompt_length:
        raise InvalidPromptError(
            f"Prompt exceeds the maximum length of "
            f"{CONFIG.max_prompt_length} characters."
        )

    return prompt


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_response(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove common invisible characters without destroying normal Markdown.
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    lines = [line.rstrip() for line in text.split("\n")]

    cleaned_lines = []

    previous_blank = False

    for line in lines:
        is_blank = not line.strip()

        if is_blank:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
        else:
            cleaned_lines.append(line)
            previous_blank = False

    return "\n".join(cleaned_lines).strip()


# ============================================================================
# CHATGPT SELECTORS
# ============================================================================

# ChatGPT's DOM can change. Keeping selectors centralized makes maintenance
# considerably easier and avoids scattering volatile selectors through the
# browser manager.

PROMPT_SELECTORS = [
    "#prompt-textarea",
    "textarea[placeholder*='Message']",
    "textarea[placeholder*='message']",
    "textarea[aria-label*='Message']",
    "textarea[aria-label*='message']",
    "textarea[data-testid*='prompt']",
    "[contenteditable='true'][role='textbox']",
    "[contenteditable='true'][data-testid*='prompt']",
    "div[contenteditable='true'][aria-label*='Message']",
    "div[contenteditable='true'][aria-label*='message']",
]

SEND_BUTTON_SELECTORS = [
    "button[data-testid='send-button']",
    "button[data-testid*='send']",
    "button[aria-label='Send prompt']",
    "button[aria-label='Send message']",
    "button[aria-label*='Send prompt']",
    "button[aria-label*='Send message']",
    "button[title='Send prompt']",
    "button[title='Send message']",
]

STOP_BUTTON_SELECTORS = [
    "button[data-testid='stop-button']",
    "button[data-testid*='stop']",
    "button[aria-label*='Stop generating']",
    "button[aria-label*='Stop generation']",
    "button[title*='Stop generating']",
]

ASSISTANT_SELECTORS = [
    "[data-message-author-role='assistant']",
    "[data-message-author-role='assistant'] div.markdown",
    "article[data-testid*='conversation-turn']",
    "div[data-testid*='conversation-turn']",
]

AUTHENTICATED_PAGE_MARKERS = [
    "#prompt-textarea",
    "textarea[placeholder*='Message']",
    "[contenteditable='true'][role='textbox']",
    "button[data-testid='send-button']",
    "button[aria-label='Send prompt']",
]


# ============================================================================
# BROWSER MANAGER
# ============================================================================

class BrowserManager:
    """
    Owns one persistent Playwright Chromium context and one reusable page.

    Flask requests must serialize access to this object using browser_lock.
    """

    def __init__(self, config: Config):
        self.config = config

        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        self._state_lock = threading.RLock()

        self.initialized = False

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def ensure_started(self) -> Page:
        with self._state_lock:
            if self._is_page_usable():
                return self._page  # type: ignore[return-value]

            self._recover_locked()

            if not self._is_page_usable():
                raise BrowserUnavailableError(
                    "The ChatGPT browser session is unavailable."
                )

            return self._page  # type: ignore[return-value]

    def _is_page_usable(self) -> bool:
        if self._context is None:
            return False

        if self._page is None:
            return False

        try:
            return not self._page.is_closed()
        except Exception:
            return False

    def _recover_locked(self) -> None:
        logger.info("Initializing/recovering persistent browser session.")

        self._close_locked()

        try:
            os.makedirs(
                self.config.user_data_dir,
                exist_ok=True,
            )

            self._playwright = sync_playwright().start()

            # launch_persistent_context keeps browser state in the configured
            # profile directory. We never inspect or export its cookies/tokens.
            self._context = (
                self._playwright.chromium.launch_persistent_context(
                    user_data_dir=self.config.user_data_dir,
                    headless=self.config.headless,
                    args=[
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                    viewport={
                        "width": 1440,
                        "height": 900,
                    },
                )
            )

            if self._context.pages:
                self._page = self._context.pages[0]
            else:
                self._page = self._context.new_page()

            self._page.set_default_timeout(10000)
            self._page.set_default_navigation_timeout(30000)

            self.initialized = True

            logger.info(
                "Persistent browser initialized successfully."
            )

        except Exception as exc:
            logger.exception("Browser initialization failed.")

            self._close_locked()

            raise BrowserUnavailableError(
                "Unable to start the Chromium browser session."
            ) from exc

    def _close_locked(self) -> None:
        self.initialized = False

        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            logger.debug(
                "Ignoring browser context close failure.",
                exc_info=True,
            )

        try:
            if self._playwright is not None:
                self._playwright.stop()
        except Exception:
            logger.debug(
                "Ignoring Playwright shutdown failure.",
                exc_info=True,
            )

        self._page = None
        self._context = None
        self._playwright = None

    def close(self) -> None:
        with self._state_lock:
            logger.info("Closing browser manager.")
            self._close_locked()

    # ---------------------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------------------

    def navigate_to_chatgpt(self, page: Page) -> None:
        try:
            current_url = page.url

            if not current_url.startswith(
                self.config.chatgpt_url.rstrip("/")
            ):
                logger.info("Navigating to configured ChatGPT URL.")

                page.goto(
                    self.config.chatgpt_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )
            else:
                # A reload gives the page a chance to recover from an old
                # partially rendered state without unnecessarily navigating
                # every request.
                if not self._page_has_prompt_input(page):
                    page.goto(
                        self.config.chatgpt_url,
                        wait_until="domcontentloaded",
                        timeout=30000,
                    )

            self._wait_until_usable(page)

        except PlaywrightTimeoutError as exc:
            raise BrowserUnavailableError(
                "ChatGPT could not be loaded within the navigation timeout."
            ) from exc

        except PlaywrightError as exc:
            raise BrowserUnavailableError(
                "The ChatGPT browser page could not be loaded."
            ) from exc

    def _wait_until_usable(self, page: Page) -> None:
        deadline = time.monotonic() + 30

        while time.monotonic() < deadline:
            if page.is_closed():
                raise BrowserUnavailableError(
                    "The ChatGPT browser page was closed."
                )

            if self._page_has_prompt_input(page):
                return

            if self._looks_like_login_page(page):
                raise AuthenticationRequiredError(
                    "ChatGPT session is not authenticated. "
                    "Authenticate the persistent browser profile first."
                )

            try:
                page.wait_for_timeout(250)
            except PlaywrightError:
                break

        if self._looks_like_login_page(page):
            raise AuthenticationRequiredError(
                "ChatGPT session is not authenticated. "
                "Authenticate the persistent browser profile first."
            )

        raise PromptInputNotFoundError(
            "ChatGPT loaded, but its prompt input could not be found. "
            "Its page structure may have changed."
        )

    # ---------------------------------------------------------------------
    # Authentication detection
    # ---------------------------------------------------------------------

    def _page_has_prompt_input(self, page: Page) -> bool:
        for selector in AUTHENTICATED_PAGE_MARKERS:
            try:
                locator = page.locator(selector).first

                if locator.count() == 0:
                    continue

                if locator.is_visible() and locator.is_enabled():
                    return True

            except Exception:
                continue

        return False

    def _looks_like_login_page(self, page: Page) -> bool:
        try:
            url = page.url.lower()

            login_url_markers = (
                "/auth/login",
                "/login",
                "/auth/",
            )

            if any(marker in url for marker in login_url_markers):
                return True

            text = page.locator("body").inner_text(
                timeout=2000
            ).lower()

            login_phrases = (
                "log in",
                "sign in",
                "continue with google",
                "continue with microsoft",
            )

            return any(phrase in text for phrase in login_phrases)

        except Exception:
            return False

    # ---------------------------------------------------------------------
    # Input detection
    # ---------------------------------------------------------------------

    def find_prompt_input(self, page: Page) -> Locator:
        for selector in PROMPT_SELECTORS:
            try:
                locator = page.locator(selector).first

                if locator.count() == 0:
                    continue

                if not locator.is_visible():
                    continue

                if not locator.is_enabled():
                    continue

                return locator

            except Exception:
                continue

        raise PromptInputNotFoundError(
            "Unable to locate the ChatGPT prompt input. "
            "The ChatGPT DOM may have changed."
        )

    # ---------------------------------------------------------------------
    # Send control detection
    # ---------------------------------------------------------------------

    def find_send_button(self, page: Page) -> Optional[Locator]:
        for selector in SEND_BUTTON_SELECTORS:
            try:
                locator = page.locator(selector).first

                if locator.count() == 0:
                    continue

                if not locator.is_visible():
                    continue

                if not locator.is_enabled():
                    continue

                return locator

            except Exception:
                continue

        return None

    def find_stop_button(self, page: Page) -> Optional[Locator]:
        for selector in STOP_BUTTON_SELECTORS:
            try:
                locator = page.locator(selector).first

                if locator.count() == 0:
                    continue

                if locator.is_visible():
                    return locator

            except Exception:
                continue

        return None

    # ---------------------------------------------------------------------
    # Conversation state
    # ---------------------------------------------------------------------

    def _assistant_messages(self, page: Page) -> list[Locator]:
        candidates = []

        for selector in ASSISTANT_SELECTORS:
            try:
                locators = page.locator(selector)

                count = locators.count()

                for index in range(count):
                    candidates.append(
                        locators.nth(index)
                    )

            except Exception:
                continue

        # Deduplicate by DOM-backed identity where possible.
        # Since Playwright Locator objects are not reliably hashable by
        # underlying DOM node, duplicate text is filtered later.
        return candidates

    def _extract_locator_text(self, locator: Locator) -> str:
        try:
            return normalize_response(
                locator.inner_text(timeout=3000)
            )
        except Exception:
            try:
                return normalize_response(
                    locator.text_content(timeout=3000) or ""
                )
            except Exception:
                return ""

    def _get_latest_assistant_text(
        self,
        page: Page,
    ) -> str:
        candidates = self._assistant_messages(page)

        texts = []

        for locator in candidates:
            try:
                if not locator.is_visible():
                    continue

                text = self._extract_locator_text(locator)

                if not text:
                    continue

                texts.append(text)

            except Exception:
                continue

        if not texts:
            return ""

        # Prefer the last rendered assistant message. This is intentionally
        # based on DOM order rather than a fixed message index.
        return texts[-1]

    def _get_assistant_snapshot(
        self,
        page: Page,
    ) -> tuple[int, str]:
        candidates = self._assistant_messages(page)

        visible_texts = []

        for locator in candidates:
            try:
                if not locator.is_visible():
                    continue

                text = self._extract_locator_text(locator)

                if text:
                    visible_texts.append(text)

            except Exception:
                continue

        if not visible_texts:
            return 0, ""

        return len(visible_texts), visible_texts[-1]

    # ---------------------------------------------------------------------
    # Prompt submission
    # ---------------------------------------------------------------------

    def send_prompt(
        self,
        page: Page,
        prompt: str,
    ) -> None:
        input_locator = self.find_prompt_input(page)

        try:
            input_locator.click(timeout=5000)

            # Fill works for both textarea-like inputs and many contenteditable
            # ChatGPT input implementations.
            input_locator.fill(prompt)

        except PlaywrightError as exc:
            raise PromptInputNotFoundError(
                "The ChatGPT prompt input was found but could not be filled."
            ) from exc

        # Verify that something was actually inserted.
        try:
            current_value = input_locator.input_value(timeout=2000)

            if not current_value.strip():
                raise PromptInputNotFoundError(
                    "The prompt could not be entered into ChatGPT."
                )

        except PlaywrightError:
            # contenteditable elements do not expose input_value().
            try:
                visible_text = normalize_response(
                    input_locator.inner_text(timeout=2000)
                )

                if not visible_text:
                    raise PromptInputNotFoundError(
                        "The prompt could not be entered into ChatGPT."
                    )

            except PlaywrightError as exc:
                raise PromptInputNotFoundError(
                    "The prompt could not be verified in ChatGPT."
                ) from exc

        send_button = self.find_send_button(page)

        if send_button is not None:
            try:
                send_button.click(timeout=5000)
                return
            except PlaywrightError:
                logger.warning(
                    "Send button was found but clicking failed; "
                    "using keyboard submission fallback."
                )

        # Keyboard fallback.
        try:
            input_locator.press("Enter")
            return
        except PlaywrightError as exc:
            raise SendButtonNotFoundError(
                "Unable to submit the ChatGPT prompt. "
                "Neither the send button nor keyboard submission worked."
            ) from exc

    # ---------------------------------------------------------------------
    # Response waiting
    # ---------------------------------------------------------------------

    def wait_for_assistant_response(
        self,
        page: Page,
        before_count: int,
        before_text: str,
    ) -> str:
        timeout = self.config.chat_timeout_seconds

        deadline = time.monotonic() + timeout

        first_meaningful_text = ""

        stable_text = ""
        stable_since = 0.0
        stable_samples = 0

        saw_new_message = False

        while time.monotonic() < deadline:
            if page.is_closed():
                raise BrowserUnavailableError(
                    "The ChatGPT browser page closed while waiting "
                    "for the response."
                )

            count, latest_text = self._get_assistant_snapshot(page)

            # A new assistant message is the preferred signal.
            if count > before_count:
                saw_new_message = True

            # If ChatGPT re-renders the message list in a way that changes the
            # count unexpectedly, a different latest text is still useful.
            if latest_text and latest_text != before_text:
                if len(latest_text) > len(before_text):
                    saw_new_message = True

            if saw_new_message and latest_text:
                if not first_meaningful_text:
                    first_meaningful_text = latest_text

                # Streaming text changes over time. We consider it complete
                # after the same non-empty text has been observed repeatedly.
                if latest_text == stable_text:
                    stable_samples += 1
                else:
                    stable_text = latest_text
                    stable_samples = 1
                    stable_since = time.monotonic()

                stop_button = self.find_stop_button(page)

                send_button = self.find_send_button(page)

                elapsed_stable = (
                    time.monotonic() - stable_since
                    if stable_since
                    else 0
                )

                # Primary completion signal:
                # response stopped changing and the stop button disappeared.
                if (
                    stable_samples >= 3
                    and elapsed_stable >= 1.0
                    and stop_button is None
                ):
                    return latest_text

                # Secondary completion signal:
                # stable response plus send control becoming available.
                if (
                    stable_samples >= 3
                    and elapsed_stable >= 1.0
                    and send_button is not None
                ):
                    return latest_text

            # Small stabilization interval only. We intentionally do not use
            # a fixed "sleep 10 seconds" strategy.
            try:
                page.wait_for_timeout(350)
            except PlaywrightError:
                break

        # If generation timed out but useful text exists, returning a partial
        # answer is preferable to silently discarding it.
        if first_meaningful_text:
            logger.warning(
                "ChatGPT response reached timeout with partial text."
            )
            return first_meaningful_text

        raise ResponseTimeoutError(
            f"ChatGPT did not produce a usable response within "
            f"{timeout} seconds."
        )

    # ---------------------------------------------------------------------
    # Public chat operation
    # ---------------------------------------------------------------------

    def chat(self, prompt: str) -> str:
        page = self.ensure_started()

        try:
            self.navigate_to_chatgpt(page)

            # Capture conversation state before sending.
            before_count, before_text = self._get_assistant_snapshot(
                page
            )

            self.send_prompt(
                page,
                prompt,
            )

            response = self.wait_for_assistant_response(
                page=page,
                before_count=before_count,
                before_text=before_text,
            )

            response = normalize_response(response)

            if not response:
                raise EmptyResponseError(
                    "ChatGPT completed the request but returned "
                    "no usable visible response."
                )

            return response

        except AuthenticationRequiredError:
            raise

        except (
            PromptInputNotFoundError,
            SendButtonNotFoundError,
            ResponseTimeoutError,
            EmptyResponseError,
        ):
            raise

        except PlaywrightTimeoutError as exc:
            raise ResponseTimeoutError(
                "The ChatGPT browser operation timed out."
            ) from exc

        except PlaywrightError as exc:
            logger.warning(
                "Playwright browser operation failed; "
                "attempting browser recovery."
            )

            # Do not expose the underlying Playwright exception to clients.
            with self._state_lock:
                self._close_locked()

            raise BrowserUnavailableError(
                "The ChatGPT browser session encountered an internal "
                "browser error and was reset."
            ) from exc

        except Exception as exc:
            logger.exception(
                "Unexpected browser manager failure."
            )

            raise BrowserUnavailableError(
                "The ChatGPT browser session failed unexpectedly."
            ) from exc


# ============================================================================
# GLOBAL BROWSER STATE
# ============================================================================

browser_manager = BrowserManager(CONFIG)

# The browser profile and page are shared resources. A single request must
# own them at a time. This prevents two simultaneous Flask requests from
# typing into the same ChatGPT conversation.
browser_lock = threading.Lock()


# ============================================================================
# ERROR RESPONSE
# ============================================================================

def error_response(
    message: str,
    status_code: int,
):
    return (
        jsonify(
            {
                "error": message,
            }
        ),
        status_code,
    )


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        max_prompt_length=CONFIG.max_prompt_length,
    )


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "chatgpt-web-automation",
            "browser_initialized": browser_manager.initialized,
        }
    )


@app.post("/api/chat")
def api_chat():
    request_id = os.urandom(6).hex()

    logger.info(
        "Chat request started | request_id=%s",
        request_id,
    )

    if not request.is_json:
        return error_response(
            "Content-Type must be application/json.",
            400,
        )

    try:
        payload = request.get_json(
            silent=False,
        )

    except Exception:
        return error_response(
            "Request body contains invalid JSON.",
            400,
        )

    try:
        prompt = validate_prompt(payload)

    except InvalidPromptError as exc:
        return error_response(
            str(exc),
            400,
        )

    logger.info(
        "Prompt accepted | request_id=%s | characters=%d",
        request_id,
        len(prompt),
    )

    acquired = browser_lock.acquire(
        timeout=CONFIG.browser_lock_timeout_seconds
    )

    if not acquired:
        logger.warning(
            "Browser lock timeout | request_id=%s",
            request_id,
        )

        return error_response(
            "The ChatGPT browser is currently busy processing another "
            "request. Please try again shortly.",
            409,
        )

    started = time.monotonic()

    try:
        reply = browser_manager.chat(prompt)

        elapsed = time.monotonic() - started

        logger.info(
            "Chat request completed | request_id=%s | elapsed=%.2fs",
            request_id,
            elapsed,
        )

        return jsonify(
            {
                "reply": reply,
            }
        )

    except InvalidPromptError as exc:
        return error_response(
            str(exc),
            400,
        )

    except AuthenticationRequiredError as exc:
        logger.warning(
            "Authentication required | request_id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            401,
        )

    except BrowserBusyError as exc:
        return error_response(
            str(exc),
            409,
        )

    except (
        PromptInputNotFoundError,
        SendButtonNotFoundError,
    ) as exc:
        logger.warning(
            "ChatGPT input failure | request_id=%s | type=%s",
            request_id,
            type(exc).__name__,
        )

        return error_response(
            str(exc),
            503,
        )

    except ResponseTimeoutError as exc:
        logger.warning(
            "ChatGPT response timeout | request_id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            504,
        )

    except EmptyResponseError as exc:
        logger.warning(
            "Empty ChatGPT response | request_id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            502,
        )

    except BrowserUnavailableError as exc:
        logger.error(
            "Browser unavailable | request_id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            503,
        )

    except Exception:
        # Never send stack traces, environment values, cookies, browser
        # details, or other internal information to the client.
        logger.exception(
            "Unhandled API failure | request_id=%s",
            request_id,
        )

        return error_response(
            "An unexpected internal error occurred while processing "
            "the ChatGPT request.",
            500,
        )

    finally:
        browser_lock.release()


# ============================================================================
# FLASK ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_too_large(_error):
    return error_response(
        "Request body is too large.",
        413,
    )


@app.errorhandler(404)
def not_found(_error):
    return error_response(
        "Endpoint not found.",
        404,
    )


@app.errorhandler(405)
def method_not_allowed(_error):
    return error_response(
        "HTTP method is not allowed for this endpoint.",
        405,
    )


@app.errorhandler(Exception)
def handle_unexpected_flask_error(error):
    logger.exception(
        "Unhandled Flask exception: %s",
        type(error).__name__,
    )

    return error_response(
        "An unexpected server error occurred.",
        500,
    )


# ============================================================================
# CLEAN SHUTDOWN
# ============================================================================

@atexit.register
def shutdown_browser():
    try:
        browser_manager.close()
    except Exception:
        logger.debug(
            "Browser shutdown encountered an error.",
            exc_info=True,
        )


# ============================================================================
# FRONTEND
# ============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <meta
        name="description"
        content="ChatGPT Web Automation Gateway"
    >

    <title>ChatGPT Web Automation Gateway</title>

    <script src="https://cdn.tailwindcss.com"></script>

    <script>
        tailwind.config = {
            darkMode: "class"
        };
    </script>
</head>

<body
    class="min-h-screen bg-slate-100 text-slate-900
           dark:bg-slate-950 dark:text-slate-100"
>
    <main class="min-h-screen flex items-center justify-center p-4">
        <section class="w-full max-w-4xl">

            <div
                class="overflow-hidden rounded-3xl border
                       border-slate-200 bg-white shadow-2xl
                       dark:border-slate-800 dark:bg-slate-900"
            >

                <!-- Header -->
                <header
                    class="border-b border-slate-200 p-6
                           dark:border-slate-800 sm:p-8"
                >
                    <div
                        class="flex flex-col gap-4
                               sm:flex-row sm:items-center
                               sm:justify-between"
                    >
                        <div>
                            <div
                                class="mb-2 inline-flex items-center gap-2
                                       rounded-full bg-emerald-100 px-3 py-1
                                       text-xs font-semibold text-emerald-700
                                       dark:bg-emerald-950
                                       dark:text-emerald-300"
                            >
                                <span
                                    id="statusDot"
                                    class="h-2 w-2 rounded-full
                                           bg-emerald-500"
                                ></span>

                                <span id="statusText">
                                    Ready
                                </span>
                            </div>

                            <h1
                                class="text-2xl font-bold tracking-tight
                                       sm:text-3xl"
                            >
                                ChatGPT Web Automation Gateway
                            </h1>

                            <p
                                class="mt-2 max-w-2xl text-sm
                                       text-slate-600
                                       dark:text-slate-400"
                            >
                                Send prompts through your authorized,
                                persistent ChatGPT browser session.
                            </p>
                        </div>

                        <button
                            id="themeButton"
                            type="button"
                            class="rounded-xl border border-slate-300
                                   px-4 py-2 text-sm font-medium
                                   transition hover:bg-slate-100
                                   dark:border-slate-700
                                   dark:hover:bg-slate-800"
                            aria-label="Toggle color theme"
                        >
                            Toggle theme
                        </button>
                    </div>
                </header>

                <!-- Content -->
                <div class="p-6 sm:p-8">

                    <form id="chatForm" novalidate>

                        <label
                            for="prompt"
                            class="mb-2 block text-sm font-semibold"
                        >
                            Prompt
                        </label>

                        <textarea
                            id="prompt"
                            name="prompt"
                            rows="8"
                            maxlength="{{ max_prompt_length }}"
                            autocomplete="off"
                            spellcheck="true"
                            placeholder="Ask ChatGPT something..."
                            class="w-full resize-y rounded-2xl border
                                   border-slate-300 bg-slate-50 p-4
                                   text-sm leading-6 outline-none
                                   transition
                                   focus:border-slate-500
                                   focus:ring-4 focus:ring-slate-200
                                   dark:border-slate-700
                                   dark:bg-slate-950
                                   dark:focus:border-slate-500
                                   dark:focus:ring-slate-800"
                            aria-describedby="promptHelp counter"
                            required
                        ></textarea>

                        <div
                            class="mt-2 flex flex-col gap-2 text-xs
                                   text-slate-500 sm:flex-row
                                   sm:items-center sm:justify-between
                                   dark:text-slate-400"
                        >
                            <span id="promptHelp">
                                Ctrl + Enter to send
                            </span>

                            <span id="counter">
                                0 / {{ max_prompt_length }}
                            </span>
                        </div>

                        <div class="mt-5 flex justify-end">
                            <button
                                id="sendButton"
                                type="submit"
                                class="inline-flex min-w-32 items-center
                                       justify-center gap-2 rounded-xl
                                       bg-slate-900 px-5 py-3 text-sm
                                       font-semibold text-white
                                       shadow-lg transition
                                       hover:bg-slate-700
                                       disabled:cursor-not-allowed
                                       disabled:opacity-50
                                       dark:bg-white
                                       dark:text-slate-900
                                       dark:hover:bg-slate-200"
                            >
                                <span id="sendLabel">
                                    Send
                                </span>

                                <svg
                                    id="loadingSpinner"
                                    class="hidden h-4 w-4 animate-spin"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    aria-hidden="true"
                                >
                                    <circle
                                        cx="12"
                                        cy="12"
                                        r="9"
                                        stroke="currentColor"
                                        stroke-width="3"
                                        opacity=".25"
                                    ></circle>

                                    <path
                                        d="M21 12a9 9 0 0 0-9-9"
                                        stroke="currentColor"
                                        stroke-width="3"
                                        stroke-linecap="round"
                                    ></path>
                                </svg>
                            </button>
                        </div>
                    </form>

                    <!-- Error -->
                    <div
                        id="errorArea"
                        class="mt-6 hidden rounded-2xl border
                               border-red-200 bg-red-50 p-4
                               text-sm text-red-800
                               dark:border-red-900
                               dark:bg-red-950/40
                               dark:text-red-200"
                        role="alert"
                    >
                        <div class="font-semibold">
                            Request failed
                        </div>

                        <div
                            id="errorMessage"
                            class="mt-1 whitespace-pre-wrap"
                        ></div>
                    </div>

                    <!-- Response -->
                    <section
                        class="mt-6"
                        aria-labelledby="responseHeading"
                    >
                        <div
                            class="mb-2 flex items-center
                                   justify-between"
                        >
                            <h2
                                id="responseHeading"
                                class="text-sm font-semibold"
                            >
                                Response
                            </h2>

                            <button
                                id="copyButton"
                                type="button"
                                class="rounded-lg border
                                       border-slate-300 px-3 py-1.5
                                       text-xs font-medium
                                       transition
                                       hover:bg-slate-100
                                       dark:border-slate-700
                                       dark:hover:bg-slate-800"
                            >
                                Copy
                            </button>
                        </div>

                        <div
                            id="responseArea"
                            class="min-h-48 rounded-2xl border
                                   border-slate-200 bg-slate-50
                                   p-5 text-sm leading-7
                                   whitespace-pre-wrap break-words
                                   dark:border-slate-800
                                   dark:bg-slate-950"
                            aria-live="polite"
                        >
                            <span
                                class="text-slate-400"
                            >
                                The ChatGPT response will appear here.
                            </span>
                        </div>
                    </section>
                </div>
            </div>

            <p
                class="mt-4 text-center text-xs text-slate-500
                       dark:text-slate-500"
            >
                Uses an authorized persistent browser session.
                No ChatGPT credentials are collected by this application.
            </p>
        </section>
    </main>

<script>
(() => {
    "use strict";

    const form = document.getElementById("chatForm");
    const promptInput = document.getElementById("prompt");
    const sendButton = document.getElementById("sendButton");
    const sendLabel = document.getElementById("sendLabel");
    const spinner = document.getElementById("loadingSpinner");

    const responseArea = document.getElementById("responseArea");

    const errorArea = document.getElementById("errorArea");
    const errorMessage = document.getElementById("errorMessage");

    const counter = document.getElementById("counter");

    const statusText = document.getElementById("statusText");
    const statusDot = document.getElementById("statusDot");

    const themeButton = document.getElementById("themeButton");
    const copyButton = document.getElementById("copyButton");

    const maxLength = Number(
        {{ max_prompt_length | tojson }}
    );

    let requestRunning = false;

    // ---------------------------------------------------------------
    // Theme
    // ---------------------------------------------------------------

    function applyTheme(theme) {
        if (theme === "dark") {
            document.documentElement.classList.add("dark");
        } else {
            document.documentElement.classList.remove("dark");
        }
    }

    const savedTheme = localStorage.getItem("gateway-theme");

    if (savedTheme === "dark" || savedTheme === "light") {
        applyTheme(savedTheme);
    } else {
        applyTheme(
            window.matchMedia &&
            window.matchMedia("(prefers-color-scheme: dark)").matches
                ? "dark"
                : "light"
        );
    }

    themeButton.addEventListener("click", () => {
        const isDark =
            document.documentElement.classList.contains("dark");

        const nextTheme = isDark ? "light" : "dark";

        applyTheme(nextTheme);

        localStorage.setItem(
            "gateway-theme",
            nextTheme
        );
    });

    // ---------------------------------------------------------------
    // Character counter
    // ---------------------------------------------------------------

    function updateCounter() {
        const length = promptInput.value.length;

        counter.textContent =
            `${length.toLocaleString()} / ${maxLength.toLocaleString()}`;

        if (length >= maxLength) {
            counter.classList.add(
                "font-semibold",
                "text-red-600"
            );
        } else {
            counter.classList.remove(
                "font-semibold",
                "text-red-600"
            );
        }
    }

    promptInput.addEventListener(
        "input",
        updateCounter
    );

    updateCounter();

    // ---------------------------------------------------------------
    // UI state
    // ---------------------------------------------------------------

    function setRunning(running) {
        requestRunning = running;

        sendButton.disabled = running;
        promptInput.disabled = running;

        if (running) {
            sendLabel.textContent = "Sending...";
            spinner.classList.remove("hidden");

            statusText.textContent = "Working";
            statusDot.classList.remove(
                "bg-emerald-500"
            );
            statusDot.classList.add(
                "bg-amber-500"
            );
        } else {
            sendLabel.textContent = "Send";
            spinner.classList.add("hidden");

            statusText.textContent = "Ready";
            statusDot.classList.remove(
                "bg-amber-500"
            );
            statusDot.classList.add(
                "bg-emerald-500"
            );
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorArea.classList.remove("hidden");
    }

    function clearError() {
        errorMessage.textContent = "";
        errorArea.classList.add("hidden");
    }

    function showResponse(text) {
        responseArea.textContent = text || "Empty response.";
    }

    // ---------------------------------------------------------------
    // API
    // ---------------------------------------------------------------

    async function submitPrompt() {
        if (requestRunning) {
            return;
        }

        const prompt = promptInput.value;

        if (!prompt.trim()) {
            showError("Please enter a prompt.");
            promptInput.focus();
            return;
        }

        if (prompt.length > maxLength) {
            showError(
                `Prompt exceeds ${maxLength.toLocaleString()} characters.`
            );
            promptInput.focus();
            return;
        }

        clearError();
        setRunning(true);

        try {
            const response = await fetch(
                "/api/chat",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    body: JSON.stringify({
                        prompt: prompt
                    })
                }
            );

            let data;

            try {
                data = await response.json();
            } catch (_jsonError) {
                throw new Error(
                    "The server returned an invalid JSON response."
                );
            }

            if (!response.ok) {
                throw new Error(
                    data &&
                    typeof data.error === "string"
                        ? data.error
                        : `Request failed with HTTP ${response.status}.`
                );
            }

            if (
                !data ||
                typeof data.reply !== "string"
            ) {
                throw new Error(
                    "The server returned an invalid ChatGPT response."
                );
            }

            showResponse(data.reply);

        } catch (error) {
            if (
                error instanceof TypeError
            ) {
                showError(
                    "Network error. Make sure the Flask server is running."
                );
            } else {
                showError(
                    error && error.message
                        ? error.message
                        : "An unexpected error occurred."
                );
            }

            // Intentionally do not clear promptInput.value.
            // The user's prompt is preserved after errors.

        } finally {
            setRunning(false);
        }
    }

    form.addEventListener(
        "submit",
        (event) => {
            event.preventDefault();
            submitPrompt();
        }
    );

    promptInput.addEventListener(
        "keydown",
        (event) => {
            if (
                event.ctrlKey &&
                event.key === "Enter"
            ) {
                event.preventDefault();

                if (!requestRunning) {
                    submitPrompt();
                }
            }
        }
    );

    // ---------------------------------------------------------------
    // Copy response
    // ---------------------------------------------------------------

    copyButton.addEventListener(
        "click",
        async () => {
            const text = responseArea.textContent.trim();

            if (!text) {
                return;
            }

            try {
                await navigator.clipboard.writeText(text);

                const original =
                    copyButton.textContent;

                copyButton.textContent = "Copied";

                setTimeout(() => {
                    copyButton.textContent = original;
                }, 1200);

            } catch (_error) {
                showError(
                    "The response could not be copied. "
                    + "Your browser may block clipboard access."
                );
            }
        }
    );
})();
</script>
</body>
</html>
"""


# ============================================================================
# DEVELOPMENT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    logger.info(
        "Starting ChatGPT Web Automation Gateway "
        "| host=0.0.0.0 | port=%d | headless=%s",
        CONFIG.port,
        CONFIG.headless,
    )

    logger.info(
        "Playwright user-data directory: %s",
        CONFIG.user_data_dir,
    )

    logger.info(
        "For production on Render, prefer: "
        "gunicorn --workers 1 --threads 4 --timeout 180 app:app"
    )

    app.run(
        host="0.0.0.0",
        port=CONFIG.port,
        debug=False,
        threaded=True,
    )
