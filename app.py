"""
app.py
============================================================
JARVIS - Web Automation Gateway
============================================================

A Flask application that provides a Jarvis-branded web interface
for interacting with an already-authenticated ChatGPT web session
through Playwright.

IMPORTANT:
- This application is intended only for an account/session the
  operator is authorized to access.
- It does NOT collect ChatGPT credentials.
- It does NOT upload/import cookies.
- It does NOT expose authentication tokens.
- It does NOT solve CAPTCHAs.
- It does NOT bypass anti-bot systems.
- It does NOT expose CDP/debugging ports.
- It does NOT attempt to evade security controls.

INSTALL:
    pip install flask playwright gunicorn

Then:
    playwright install chromium

RENDER START COMMAND:
    gunicorn --workers 1 --threads 4 --timeout 180 app:app

For local development:
    set HEADLESS=false

Then manually authenticate the persistent browser profile.

ENVIRONMENT VARIABLES:
    PORT=10000
    HEADLESS=true
    CHATGPT_URL=https://chatgpt.com/
    PLAYWRIGHT_USER_DATA_DIR=./playwright_data
    CHAT_TIMEOUT_SECONDS=120
    BROWSER_LOCK_TIMEOUT_SECONDS=180
    MAX_PROMPT_LENGTH=10000
    LOG_LEVEL=INFO

RENDER STORAGE NOTE:
Render's normal filesystem can be ephemeral. If the browser profile
must survive restarts, PLAYWRIGHT_USER_DATA_DIR should point to
persistent storage configured for the deployment.

IMPORTANT GUNICORN NOTE:
Use one Gunicorn worker.

The BrowserManager is process-local. Multiple workers would create
multiple independent Playwright browser managers and profiles.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Any

from flask import (
    Flask,
    jsonify,
    render_template_string,
    request,
)

from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


# ============================================================
# CONFIGURATION
# ============================================================

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


def env_int(
    name: str,
    default: int,
    minimum: Optional[int] = None,
) -> int:
    value = os.environ.get(name)

    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except (TypeError, ValueError):
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

        headless=env_bool(
            "HEADLESS",
            True,
        ),

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


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=getattr(
        logging,
        CONFIG.log_level,
        logging.INFO,
    ),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "jarvis-web-automation"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.config.update(
    MAX_CONTENT_LENGTH=max(
        CONFIG.max_prompt_length + 4096,
        16384,
    ),
    JSON_SORT_KEYS=False,
)


# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class GatewayError(Exception):
    """Base application exception."""


class InvalidPromptError(GatewayError):
    """Invalid API prompt."""


class AuthenticationRequiredError(GatewayError):
    """Browser session is not authenticated."""


class BrowserUnavailableError(GatewayError):
    """Browser could not be started or recovered."""


class PromptInputNotFoundError(GatewayError):
    """Jarvis could not locate the remote prompt input."""


class SendButtonNotFoundError(GatewayError):
    """Remote send control could not be located."""


class ResponseTimeoutError(GatewayError):
    """Response generation exceeded the configured timeout."""


class EmptyResponseError(GatewayError):
    """No usable response was extracted."""


# ============================================================
# SECURITY HEADERS
# ============================================================

@app.after_request
def security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"

    response.headers["Permissions-Policy"] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' "
        "https://cdn.tailwindcss.com "
        "'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:;"
    )

    return response


# ============================================================
# VALIDATION
# ============================================================

def validate_prompt(payload: Any) -> str:
    if payload is None:
        raise InvalidPromptError(
            "Request body must contain JSON."
        )

    if not isinstance(payload, dict):
        raise InvalidPromptError(
            "JSON body must be an object."
        )

    if "prompt" not in payload:
        raise InvalidPromptError(
            "The 'prompt' field is required."
        )

    prompt = payload["prompt"]

    if not isinstance(prompt, str):
        raise InvalidPromptError(
            "The 'prompt' field must be a string."
        )

    if not prompt.strip():
        raise InvalidPromptError(
            "The prompt cannot be empty."
        )

    if len(prompt) > CONFIG.max_prompt_length:
        raise InvalidPromptError(
            f"Prompt exceeds the maximum length of "
            f"{CONFIG.max_prompt_length} characters."
        )

    return prompt


def normalize_response(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    lines = text.split("\n")

    output = []
    previous_blank = False

    for line in lines:
        line = line.rstrip()

        if not line.strip():
            if not previous_blank:
                output.append("")

            previous_blank = True
        else:
            output.append(line)
            previous_blank = False

    return "\n".join(output).strip()


# ============================================================
# CHATGPT DOM SELECTORS
# ============================================================
#
# These selectors are isolated here intentionally.
#
# The remote site's DOM can change. If that happens, this section
# can be updated without rewriting the Flask application.
#
# We deliberately use multiple strategies:
# - stable IDs
# - semantic roles
# - ARIA labels
# - data attributes
# - textarea/contenteditable fallbacks
#
# We do NOT rely exclusively on generated CSS class names.
# ============================================================

PROMPT_SELECTORS = [
    "#prompt-textarea",

    "textarea[placeholder*='Message']",
    "textarea[placeholder*='message']",

    "textarea[aria-label*='Message']",
    "textarea[aria-label*='message']",

    "textarea[data-testid*='prompt']",

    "textarea",

    "[contenteditable='true'][role='textbox']",

    "[contenteditable='true'][data-testid*='prompt']",

    "[contenteditable='true'][aria-label*='Message']",
    "[contenteditable='true'][aria-label*='message']",

    "[role='textbox'][contenteditable='true']",

    "[role='textbox']",
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

    "button[type='submit']",
]


STOP_BUTTON_SELECTORS = [
    "button[data-testid='stop-button']",
    "button[data-testid*='stop']",

    "button[aria-label*='Stop generating']",
    "button[aria-label*='Stop generation']",

    "button[title*='Stop generating']",
    "button[title*='Stop generation']",
]


# Primary modern message selector.
#
# The first selector is preferred because it directly indicates the
# author role instead of depending on visual styling.
ASSISTANT_SELECTORS = [
    "[data-message-author-role='assistant']",

    "[data-message-author-role='assistant'] .markdown",

    "article[data-testid*='conversation-turn']",

    "div[data-testid*='conversation-turn']",
]


# ============================================================
# BROWSER MANAGER
# ============================================================

class BrowserManager:
    """
    Owns a single persistent Chromium context and reusable page.

    Access is serialized externally by browser_lock.
    """

    def __init__(self, config: Config):
        self.config = config

        self._playwright = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        self._state_lock = threading.RLock()

        self.initialized = False

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def ensure_started(self) -> Page:
        with self._state_lock:

            if self._is_page_usable():
                return self._page  # type: ignore

            self._recover_locked()

            if not self._is_page_usable():
                raise BrowserUnavailableError(
                    "The Jarvis browser session is unavailable."
                )

            return self._page  # type: ignore

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
        logger.info(
            "Initializing/recovering persistent browser."
        )

        self._close_locked()

        try:
            os.makedirs(
                self.config.user_data_dir,
                exist_ok=True,
            )

            self._playwright = (
                sync_playwright().start()
            )

            self._context = (
                self._playwright
                .chromium
                .launch_persistent_context(
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

            self._page.set_default_timeout(
                10000
            )

            self._page.set_default_navigation_timeout(
                30000
            )

            self.initialized = True

            logger.info(
                "Persistent browser initialized."
            )

        except Exception as exc:
            logger.exception(
                "Browser initialization failed."
            )

            self._close_locked()

            raise BrowserUnavailableError(
                "Unable to start Chromium."
            ) from exc

    def _close_locked(self) -> None:
        self.initialized = False

        try:
            if self._context is not None:
                self._context.close()

        except Exception:
            logger.debug(
                "Context close failed.",
                exc_info=True,
            )

        try:
            if self._playwright is not None:
                self._playwright.stop()

        except Exception:
            logger.debug(
                "Playwright stop failed.",
                exc_info=True,
            )

        self._page = None
        self._context = None
        self._playwright = None

    def close(self) -> None:
        with self._state_lock:
            logger.info(
                "Closing Jarvis browser manager."
            )

            self._close_locked()

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    def navigate_to_chatgpt(
        self,
        page: Page,
    ) -> None:
        try:
            current_url = page.url

            target_root = (
                self.config.chatgpt_url
                .rstrip("/")
            )

            if not current_url.startswith(
                target_root
            ):
                logger.info(
                    "Navigating browser to configured "
                    "remote chat service."
                )

                page.goto(
                    self.config.chatgpt_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

            # Give client-side rendering enough time to initialize.
            self._wait_for_chat_ui(page)

        except AuthenticationRequiredError:
            raise

        except PlaywrightTimeoutError as exc:
            raise BrowserUnavailableError(
                "The remote chat page timed out while loading."
            ) from exc

        except PlaywrightError as exc:
            raise BrowserUnavailableError(
                "The remote chat page could not be loaded."
            ) from exc

    def _wait_for_chat_ui(
        self,
        page: Page,
    ) -> None:
        deadline = (
            time.monotonic() + 35
        )

        last_url = ""

        while time.monotonic() < deadline:

            if page.is_closed():
                raise BrowserUnavailableError(
                    "The browser page was closed."
                )

            try:
                last_url = page.url

                if self._find_prompt_input(
                    page,
                    raise_error=False,
                ) is not None:
                    return

            except Exception:
                pass

            if self._looks_like_auth_page(
                page
            ):
                raise AuthenticationRequiredError(
                    "The browser session is not authenticated. "
                    "Open the persistent browser profile and "
                    "authenticate manually first."
                )

            try:
                page.wait_for_timeout(300)

            except PlaywrightError:
                break

        if self._looks_like_auth_page(page):
            raise AuthenticationRequiredError(
                "The browser session is not authenticated. "
                "Authenticate the persistent browser profile first."
            )

        raise PromptInputNotFoundError(
            "Jarvis loaded the remote chat page, but could not "
            "locate its message input. "
            f"Current page: {last_url or 'unknown'}"
        )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    def _looks_like_auth_page(
        self,
        page: Page,
    ) -> bool:
        try:
            url = page.url.lower()

            if any(
                marker in url
                for marker in (
                    "/auth/login",
                    "/login",
                    "/auth/",
                )
            ):
                return True

            body_text = page.locator(
                "body"
            ).inner_text(
                timeout=1500
            ).lower()

            auth_phrases = (
                "log in",
                "sign in",
                "continue with google",
                "continue with microsoft",
            )

            return any(
                phrase in body_text
                for phrase in auth_phrases
            )

        except Exception:
            return False

    # --------------------------------------------------------
    # Prompt input detection
    # --------------------------------------------------------

    def _find_prompt_input(
        self,
        page: Page,
        raise_error: bool = True,
    ) -> Optional[Locator]:

        # First pass: known robust selectors.
        for selector in PROMPT_SELECTORS:

            try:
                locator = page.locator(
                    selector
                ).first

                if locator.count() == 0:
                    continue

                if not locator.is_visible():
                    continue

                if not locator.is_enabled():
                    continue

                return locator

            except Exception:
                continue

        # Second pass: semantic role.
        try:
            textbox = page.get_by_role(
                "textbox"
            ).last

            if (
                textbox.count() > 0
                and textbox.is_visible()
                and textbox.is_enabled()
            ):
                return textbox

        except Exception:
            pass

        # Third pass: inspect visible textareas/contenteditables.
        try:
            textarea_count = page.locator(
                "textarea:visible"
            ).count()

            if textarea_count:
                for index in range(
                    textarea_count - 1,
                    -1,
                    -1,
                ):
                    locator = page.locator(
                        "textarea:visible"
                    ).nth(index)

                    if locator.is_enabled():
                        return locator

        except Exception:
            pass

        try:
            editable_count = page.locator(
                "[contenteditable='true']:visible"
            ).count()

            if editable_count:
                for index in range(
                    editable_count - 1,
                    -1,
                    -1,
                ):
                    locator = page.locator(
                        "[contenteditable='true']:visible"
                    ).nth(index)

                    if locator.is_enabled():
                        return locator

        except Exception:
            pass

        if raise_error:
            raise PromptInputNotFoundError(
                "Unable to locate the remote message input. "
                "The remote page structure may have changed."
            )

        return None

    def find_prompt_input(
        self,
        page: Page,
    ) -> Locator:
        locator = self._find_prompt_input(
            page,
            raise_error=True,
        )

        if locator is None:
            raise PromptInputNotFoundError(
                "Message input was not found."
            )

        return locator

    # --------------------------------------------------------
    # Send button
    # --------------------------------------------------------

    def find_send_button(
        self,
        page: Page,
    ) -> Optional[Locator]:

        for selector in SEND_BUTTON_SELECTORS:

            try:
                locator = page.locator(
                    selector
                ).first

                if locator.count() == 0:
                    continue

                if not locator.is_visible():
                    continue

                if not locator.is_enabled():
                    continue

                return locator

            except Exception:
                continue

        # Semantic fallback.
        try:
            buttons = page.get_by_role(
                "button"
            )

            count = buttons.count()

            for index in range(
                count - 1,
                -1,
                -1,
            ):
                button = buttons.nth(index)

                try:
                    if not button.is_visible():
                        continue

                    if not button.is_enabled():
                        continue

                    label = (
                        button.get_attribute(
                            "aria-label"
                        )
                        or ""
                    ).lower()

                    title = (
                        button.get_attribute(
                            "title"
                        )
                        or ""
                    ).lower()

                    text = (
                        button.inner_text(
                            timeout=500
                        )
                        or ""
                    ).lower()

                    combined = (
                        f"{label} "
                        f"{title} "
                        f"{text}"
                    )

                    if "send" in combined:
                        return button

                except Exception:
                    continue

        except Exception:
            pass

        return None

    # --------------------------------------------------------
    # Stop-generation button
    # --------------------------------------------------------

    def find_stop_button(
        self,
        page: Page,
    ) -> Optional[Locator]:

        for selector in STOP_BUTTON_SELECTORS:

            try:
                locator = page.locator(
                    selector
                ).first

                if locator.count() == 0:
                    continue

                if locator.is_visible():
                    return locator

            except Exception:
                continue

        return None

    # --------------------------------------------------------
    # Conversation state
    # --------------------------------------------------------

    def _get_assistant_locators(
        self,
        page: Page,
    ) -> list[Locator]:

        results: list[Locator] = []

        for selector in ASSISTANT_SELECTORS:

            try:
                collection = page.locator(
                    selector
                )

                count = collection.count()

                for index in range(count):
                    results.append(
                        collection.nth(index)
                    )

            except Exception:
                continue

        return results

    def _locator_text(
        self,
        locator: Locator,
    ) -> str:

        try:
            return normalize_response(
                locator.inner_text(
                    timeout=1500
                )
            )

        except Exception:
            try:
                return normalize_response(
                    locator.text_content(
                        timeout=1500
                    )
                    or ""
                )

            except Exception:
                return ""

    def _get_assistant_messages(
        self,
        page: Page,
    ) -> list[str]:

        texts: list[str] = []

        for locator in self._get_assistant_locators(
            page
        ):

            try:
                if not locator.is_visible():
                    continue

                text = self._locator_text(
                    locator
                )

                if not text:
                    continue

                # Avoid duplicated representations of
                # the same message caused by fallback selectors.
                if texts and text == texts[-1]:
                    continue

                texts.append(text)

            except Exception:
                continue

        return texts

    def get_conversation_snapshot(
        self,
        page: Page,
    ) -> tuple[int, str]:

        messages = self._get_assistant_messages(
            page
        )

        if not messages:
            return 0, ""

        return (
            len(messages),
            messages[-1],
        )

    # --------------------------------------------------------
    # Fill prompt safely
    # --------------------------------------------------------

    def _fill_input(
        self,
        locator: Locator,
        prompt: str,
    ) -> None:

        try:
            locator.click(
                timeout=5000
            )

        except PlaywrightError as exc:
            raise PromptInputNotFoundError(
                "The message input could not be focused."
            ) from exc

        # Preferred path.
        try:
            locator.fill(prompt)

        except PlaywrightError:
            # Some contenteditable implementations are
            # more reliable with keyboard insertion.
            try:
                locator.press(
                    "Control+A"
                )

                locator.press(
                    "Backspace"
                )

                locator.press_sequentially(
                    prompt,
                    delay=0,
                )

            except PlaywrightError as exc:
                raise PromptInputNotFoundError(
                    "The message could not be entered."
                ) from exc

        # Verify insertion.
        try:
            value = locator.input_value(
                timeout=1500
            )

            if value.strip():
                return

        except Exception:
            pass

        try:
            text = locator.inner_text(
                timeout=1500
            )

            if text.strip():
                return

        except Exception:
            pass

        # Some editors expose textContent instead.
        try:
            text = locator.text_content(
                timeout=1500
            )

            if text and text.strip():
                return

        except Exception:
            pass

        raise PromptInputNotFoundError(
            "The prompt was not successfully inserted "
            "into the remote message editor."
        )

    # --------------------------------------------------------
    # Send prompt
    # --------------------------------------------------------

    def send_prompt(
        self,
        page: Page,
        prompt: str,
    ) -> None:

        input_locator = self.find_prompt_input(
            page
        )

        self._fill_input(
            input_locator,
            prompt,
        )

        send_button = self.find_send_button(
            page
        )

        # Preferred: click the actual send control.
        if send_button is not None:

            try:
                send_button.click(
                    timeout=5000
                )

                # Give the DOM a moment to reflect
                # the submitted state.
                try:
                    page.wait_for_timeout(250)
                except Exception:
                    pass

                return

            except PlaywrightError:
                logger.warning(
                    "Send button click failed; "
                    "using keyboard fallback."
                )

        # Keyboard fallback.
        #
        # We use Enter only when no usable send button exists.
        # This prevents accidental double submission.
        try:
            input_locator.press(
                "Enter"
            )

            return

        except PlaywrightError as exc:
            raise SendButtonNotFoundError(
                "Unable to submit the message. "
                "The send control could not be located "
                "and keyboard submission failed."
            ) from exc

    # --------------------------------------------------------
    # Response waiting
    # --------------------------------------------------------

    def wait_for_response(
        self,
        page: Page,
        before_count: int,
        before_text: str,
    ) -> str:

        deadline = (
            time.monotonic()
            + self.config.chat_timeout_seconds
        )

        saw_new_response = False

        latest_text = ""

        stable_text = ""

        stable_since = 0.0

        stable_samples = 0

        first_useful_text = ""

        while time.monotonic() < deadline:

            if page.is_closed():
                raise BrowserUnavailableError(
                    "The browser page closed while "
                    "waiting for the response."
                )

            messages = self._get_assistant_messages(
                page
            )

            count = len(messages)

            current_text = (
                messages[-1]
                if messages
                else ""
            )

            # Primary signal:
            # a new assistant message exists.
            if count > before_count:
                saw_new_response = True

            # Secondary signal:
            # current last message changed from
            # the pre-submission message.
            if (
                current_text
                and current_text != before_text
            ):
                saw_new_response = True

            if (
                saw_new_response
                and current_text
            ):

                latest_text = current_text

                if not first_useful_text:
                    first_useful_text = (
                        current_text
                    )

                if current_text == stable_text:
                    stable_samples += 1

                else:
                    stable_text = current_text
                    stable_samples = 1
                    stable_since = (
                        time.monotonic()
                    )

                stop_button = (
                    self.find_stop_button(
                        page
                    )
                )

                send_button = (
                    self.find_send_button(
                        page
                    )
                )

                stable_duration = (
                    time.monotonic()
                    - stable_since
                    if stable_since
                    else 0
                )

                # Strong completion signal.
                if (
                    stable_samples >= 3
                    and stable_duration >= 1.0
                    and stop_button is None
                ):
                    return normalize_response(
                        latest_text
                    )

                # Secondary completion signal.
                if (
                    stable_samples >= 3
                    and stable_duration >= 1.0
                    and send_button is not None
                ):
                    return normalize_response(
                        latest_text
                    )

            try:
                page.wait_for_timeout(
                    300
                )

            except PlaywrightError:
                break

        # If timeout occurred after useful streamed text
        # appeared, return that partial response rather
        # than pretending there was no response.
        if first_useful_text:
            logger.warning(
                "Response timeout reached with "
                "usable partial response."
            )

            return normalize_response(
                first_useful_text
            )

        raise ResponseTimeoutError(
            "The response was not produced within "
            f"{self.config.chat_timeout_seconds} seconds."
        )

    # --------------------------------------------------------
    # Public chat operation
    # --------------------------------------------------------

    def chat(
        self,
        prompt: str,
    ) -> str:

        page = self.ensure_started()

        try:
            self.navigate_to_chatgpt(
                page
            )

            before_count, before_text = (
                self.get_conversation_snapshot(
                    page
                )
            )

            self.send_prompt(
                page,
                prompt,
            )

            response = self.wait_for_response(
                page=page,
                before_count=before_count,
                before_text=before_text,
            )

            response = normalize_response(
                response
            )

            if not response:
                raise EmptyResponseError(
                    "The remote service returned "
                    "an empty response."
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
                "The browser operation timed out."
            ) from exc

        except PlaywrightError as exc:

            logger.warning(
                "Playwright failure. Resetting browser."
            )

            with self._state_lock:
                self._close_locked()

            raise BrowserUnavailableError(
                "The browser session encountered "
                "an internal error and was reset."
            ) from exc

        except Exception as exc:

            logger.exception(
                "Unexpected browser manager failure."
            )

            raise BrowserUnavailableError(
                "The browser session failed unexpectedly."
            ) from exc


# ============================================================
# GLOBAL BROWSER STATE
# ============================================================

browser_manager = BrowserManager(
    CONFIG
)

# Critical:
# Only one request may interact with the browser at a time.
browser_lock = threading.Lock()


# ============================================================
# JSON ERROR HELPER
# ============================================================

def error_response(
    message: str,
    status_code: int,
):
    return (
        jsonify(
            {
                "error": message
            }
        ),
        status_code,
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        max_prompt_length=CONFIG.max_prompt_length,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "jarvis-web-automation",
            "browser_initialized": (
                browser_manager.initialized
            ),
        }
    )


# ============================================================
# CHAT API
# ============================================================

@app.post("/api/chat")
def api_chat():

    request_id = os.urandom(
        6
    ).hex()

    logger.info(
        "Request started | id=%s",
        request_id,
    )

    if not request.is_json:
        return error_response(
            "Content-Type must be application/json.",
            400,
        )

    try:
        payload = request.get_json(
            silent=False
        )

    except Exception:
        return error_response(
            "Request body contains invalid JSON.",
            400,
        )

    try:
        prompt = validate_prompt(
            payload
        )

    except InvalidPromptError as exc:
        return error_response(
            str(exc),
            400,
        )

    logger.info(
        "Prompt accepted | id=%s | characters=%d",
        request_id,
        len(prompt),
    )

    acquired = browser_lock.acquire(
        timeout=CONFIG.browser_lock_timeout_seconds
    )

    if not acquired:
        logger.warning(
            "Browser lock timeout | id=%s",
            request_id,
        )

        return error_response(
            "Jarvis is currently processing another request. "
            "Please try again shortly.",
            409,
        )

    started = time.monotonic()

    try:

        reply = browser_manager.chat(
            prompt
        )

        elapsed = (
            time.monotonic()
            - started
        )

        logger.info(
            "Request completed | id=%s | elapsed=%.2fs",
            request_id,
            elapsed,
        )

        return jsonify(
            {
                "reply": reply
            }
        )

    except AuthenticationRequiredError as exc:

        logger.warning(
            "Authentication required | id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            401,
        )

    except (
        PromptInputNotFoundError,
        SendButtonNotFoundError,
    ) as exc:

        logger.warning(
            "Input failure | id=%s | type=%s",
            request_id,
            type(exc).__name__,
        )

        return error_response(
            str(exc),
            503,
        )

    except ResponseTimeoutError as exc:

        logger.warning(
            "Response timeout | id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            504,
        )

    except EmptyResponseError as exc:

        logger.warning(
            "Empty response | id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            502,
        )

    except BrowserUnavailableError as exc:

        logger.error(
            "Browser unavailable | id=%s",
            request_id,
        )

        return error_response(
            str(exc),
            503,
        )

    except Exception:

        logger.exception(
            "Unexpected API error | id=%s",
            request_id,
        )

        return error_response(
            "An unexpected internal error occurred.",
            500,
        )

    finally:
        browser_lock.release()


# ============================================================
# FLASK ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):
    return error_response(
        "Request body is too large.",
        413,
    )


@app.errorhandler(404)
def route_not_found(_error):
    return error_response(
        "Endpoint not found.",
        404,
    )


@app.errorhandler(405)
def method_not_allowed(_error):
    return error_response(
        "HTTP method is not allowed.",
        405,
    )


@app.errorhandler(Exception)
def unexpected_flask_error(error):

    logger.exception(
        "Unhandled Flask error | type=%s",
        type(error).__name__,
    )

    return error_response(
        "An unexpected server error occurred.",
        500,
    )


# ============================================================
# CLEAN SHUTDOWN
# ============================================================

@atexit.register
def shutdown_browser():

    try:
        browser_manager.close()

    except Exception:
        logger.debug(
            "Browser shutdown error.",
            exc_info=True,
        )


# ============================================================
# JARVIS FRONTEND
# ============================================================

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
        content="Jarvis intelligent assistant"
    >

    <title>Jarvis</title>

    <script src="https://cdn.tailwindcss.com"></script>

    <script>
        tailwind.config = {
            darkMode: "class"
        };
    </script>

</head>


<body
    class="
        min-h-screen
        bg-slate-950
        text-white
        selection:bg-blue-500/30
    "
>

<main
    class="
        min-h-screen
        flex
        items-center
        justify-center
        p-4
    "
>

<section
    class="
        w-full
        max-w-5xl
        overflow-hidden
        rounded-3xl
        border
        border-slate-800
        bg-slate-900/90
        shadow-2xl
    "
>

<!-- =======================================================
     HEADER
======================================================= -->

<header
    class="
        border-b
        border-slate-800
        bg-slate-950/60
        px-6
        py-6
        sm:px-8
    "
>

<div
    class="
        flex
        flex-col
        gap-4
        sm:flex-row
        sm:items-center
        sm:justify-between
    "
>

<div>

<div
    class="
        flex
        items-center
        gap-3
    "
>

<!-- Jarvis logo -->
<div
    class="
        flex
        h-12
        w-12
        items-center
        justify-center
        rounded-2xl
        bg-gradient-to-br
        from-blue-500
        to-violet-600
        text-2xl
        shadow-lg
        shadow-blue-500/20
    "
    aria-hidden="true"
>
    ✦
</div>

<div>

<h1
    class="
        text-3xl
        font-bold
        tracking-tight
        bg-gradient-to-r
        from-blue-400
        to-violet-400
        bg-clip-text
        text-transparent
    "
>
    Jarvis
</h1>

<p
    class="
        text-xs
        text-slate-400
    "
>
    Intelligent Assistant
</p>

</div>

</div>

</div>


<div
    class="
        inline-flex
        w-fit
        items-center
        gap-2
        rounded-full
        border
        border-emerald-900
        bg-emerald-950/40
        px-3
        py-1.5
        text-xs
        font-medium
        text-emerald-300
    "
>

<span
    id="statusDot"
    class="
        h-2
        w-2
        rounded-full
        bg-emerald-400
    "
></span>

<span id="statusText">
    Ready
</span>

</div>

</div>

</header>


<!-- =======================================================
     BODY
======================================================= -->

<div
    class="
        p-6
        sm:p-8
    "
>

<form
    id="chatForm"
    novalidate
>


<label
    for="prompt"
    class="
        mb-3
        block
        text-sm
        font-semibold
        text-slate-200
    "
>
    Message Jarvis
</label>


<div
    class="
        overflow-hidden
        rounded-2xl
        border
        border-slate-700
        bg-slate-950
        shadow-inner
        transition
        focus-within:border-blue-500
        focus-within:ring-4
        focus-within:ring-blue-500/10
    "
>

<textarea
    id="prompt"
    name="prompt"
    rows="8"
    maxlength="{{ max_prompt_length }}"
    autocomplete="off"
    spellcheck="true"
    placeholder="Type your message here..."
    class="
        block
        w-full
        resize-y
        border-0
        bg-transparent
        p-5
        text-sm
        leading-7
        text-white
        outline-none
        placeholder:text-slate-600
    "
    required
></textarea>

</div>


<div
    class="
        mt-3
        flex
        flex-col
        gap-2
        text-xs
        text-slate-500
        sm:flex-row
        sm:items-center
        sm:justify-between
    "
>

<span>
    Ctrl + Enter to send
</span>

<span id="counter">
    0 / {{ max_prompt_length }}
</span>

</div>


<div
    class="
        mt-5
        flex
        justify-end
    "
>

<button
    id="sendButton"
    type="submit"
    class="
        inline-flex
        min-w-32
        items-center
        justify-center
        gap-2
        rounded-xl
        bg-gradient-to-r
        from-blue-600
        to-violet-600
        px-6
        py-3
        text-sm
        font-semibold
        text-white
        shadow-lg
        shadow-blue-500/20
        transition
        hover:scale-[1.01]
        hover:from-blue-500
        hover:to-violet-500
        disabled:cursor-not-allowed
        disabled:opacity-50
        disabled:hover:scale-100
    "
>

<span id="sendLabel">
    Send
</span>


<svg
    id="loadingSpinner"
    class="
        hidden
        h-4
        w-4
        animate-spin
    "
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


<!-- =======================================================
     ERROR
======================================================= -->

<div
    id="errorArea"
    class="
        mt-6
        hidden
        rounded-2xl
        border
        border-red-900
        bg-red-950/40
        p-4
        text-sm
        text-red-200
    "
    role="alert"
>

<div
    class="
        font-semibold
        text-red-300
    "
>
    Request failed
</div>

<div
    id="errorMessage"
    class="mt-1 whitespace-pre-wrap"
></div>

</div>


<!-- =======================================================
     RESPONSE
======================================================= -->

<section
    class="mt-7"
    aria-labelledby="responseHeading"
>

<div
    class="
        mb-3
        flex
        items-center
        justify-between
    "
>

<h2
    id="responseHeading"
    class="
        text-sm
        font-semibold
        text-slate-200
    "
>
    Jarvis response
</h2>


<button
    id="copyButton"
    type="button"
    class="
        rounded-lg
        border
        border-slate-700
        px-3
        py-1.5
        text-xs
        font-medium
        text-slate-300
        transition
        hover:bg-slate-800
        disabled:cursor-not-allowed
        disabled:opacity-40
    "
    disabled
>
    Copy
</button>

</div>


<div
    id="responseArea"
    class="
        min-h-56
        rounded-2xl
        border
        border-slate-800
        bg-slate-950
        p-6
        text-sm
        leading-7
        text-slate-200
        whitespace-pre-wrap
        break-words
    "
    aria-live="polite"
>

<span
    id="emptyResponse"
    class="text-slate-600"
>
    The Jarvis response will appear here.
</span>

</div>

</section>

</div>

</section>

</main>


<script>
(() => {

    "use strict";


    // ========================================================
    // ELEMENTS
    // ========================================================

    const form =
        document.getElementById("chatForm");

    const promptInput =
        document.getElementById("prompt");

    const sendButton =
        document.getElementById("sendButton");

    const sendLabel =
        document.getElementById("sendLabel");

    const spinner =
        document.getElementById("loadingSpinner");

    const responseArea =
        document.getElementById("responseArea");

    const emptyResponse =
        document.getElementById("emptyResponse");

    const errorArea =
        document.getElementById("errorArea");

    const errorMessage =
        document.getElementById("errorMessage");

    const counter =
        document.getElementById("counter");

    const statusText =
        document.getElementById("statusText");

    const statusDot =
        document.getElementById("statusDot");

    const copyButton =
        document.getElementById("copyButton");


    const maxLength =
        Number(
            {{ max_prompt_length | tojson }}
        );


    let requestRunning = false;


    // ========================================================
    // COUNTER
    // ========================================================

    function updateCounter() {

        const length =
            promptInput.value.length;

        counter.textContent =
            `${length.toLocaleString()} / `
            + `${maxLength.toLocaleString()}`;

        if (length >= maxLength) {

            counter.classList.add(
                "text-red-400",
                "font-semibold"
            );

        } else {

            counter.classList.remove(
                "text-red-400",
                "font-semibold"
            );
        }
    }


    promptInput.addEventListener(
        "input",
        updateCounter
    );


    updateCounter();


    // ========================================================
    // STATUS
    // ========================================================

    function setRunning(running) {

        requestRunning = running;

        sendButton.disabled =
            running;

        promptInput.disabled =
            running;


        if (running) {

            sendLabel.textContent =
                "Thinking...";

            spinner.classList.remove(
                "hidden"
            );

            statusText.textContent =
                "Working";

            statusDot.classList.remove(
                "bg-emerald-400"
            );

            statusDot.classList.add(
                "bg-blue-400",
                "animate-pulse"
            );

        } else {

            sendLabel.textContent =
                "Send";

            spinner.classList.add(
                "hidden"
            );

            statusText.textContent =
                "Ready";

            statusDot.classList.remove(
                "bg-blue-400",
                "animate-pulse"
            );

            statusDot.classList.add(
                "bg-emerald-400"
            );
        }
    }


    // ========================================================
    // ERROR
    // ========================================================

    function clearError() {

        errorMessage.textContent =
            "";

        errorArea.classList.add(
            "hidden"
        );
    }


    function showError(message) {

        errorMessage.textContent =
            message;

        errorArea.classList.remove(
            "hidden"
        );
    }


    // ========================================================
    // RESPONSE
    // ========================================================

    function clearResponse() {

        responseArea.textContent =
            "";

        const placeholder =
            document.createElement(
                "span"
            );

        placeholder.id =
            "emptyResponse";

        placeholder.className =
            "text-slate-600";

        placeholder.textContent =
            "The Jarvis response will appear here.";

        responseArea.appendChild(
            placeholder
        );

        copyButton.disabled =
            true;
    }


    function showResponse(text) {

        responseArea.textContent =
            text;

        copyButton.disabled =
            !text.trim();
    }


    // ========================================================
    // SUBMIT
    // ========================================================

    async function submitPrompt() {

        if (requestRunning) {
            return;
        }


        const prompt =
            promptInput.value;


        if (!prompt.trim()) {

            showError(
                "Please enter a message."
            );

            promptInput.focus();

            return;
        }


        if (prompt.length > maxLength) {

            showError(
                `Message exceeds `
                + `${maxLength.toLocaleString()} `
                + `characters.`
            );

            promptInput.focus();

            return;
        }


        clearError();

        clearResponse();

        setRunning(true);


        try {

            const response =
                await fetch(
                    "/api/chat",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",

                            "Accept":
                                "application/json"
                        },

                        body: JSON.stringify({
                            prompt: prompt
                        })
                    }
                );


            let data;


            try {

                data =
                    await response.json();

            } catch (_jsonError) {

                throw new Error(
                    "The server returned invalid JSON."
                );
            }


            if (!response.ok) {

                throw new Error(
                    data &&
                    typeof data.error === "string"

                        ? data.error

                        : (
                            `Request failed with `
                            + `HTTP ${response.status}.`
                        )
                );
            }


            if (
                !data ||
                typeof data.reply !== "string"
            ) {

                throw new Error(
                    "The server returned an invalid response."
                );
            }


            showResponse(
                data.reply
            );


        } catch (error) {

            if (
                error instanceof TypeError
            ) {

                showError(
                    "Network error. "
                    + "Check that the Jarvis server is running."
                );

            } else {

                showError(
                    error &&
                    error.message

                        ? error.message

                        : "An unexpected error occurred."
                );
            }

            // The entered message is deliberately preserved.


        } finally {

            setRunning(false);
        }
    }


    // ========================================================
    // FORM
    // ========================================================

    form.addEventListener(
        "submit",
        (event) => {

            event.preventDefault();

            submitPrompt();
        }
    );


    // ========================================================
    // CTRL + ENTER
    // ========================================================

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


    // ========================================================
    // COPY
    // ========================================================

    copyButton.addEventListener(
        "click",
        async () => {

            const text =
                responseArea.textContent.trim();


            if (!text) {
                return;
            }


            try {

                await navigator.clipboard.writeText(
                    text
                );


                const original =
                    copyButton.textContent;


                copyButton.textContent =
                    "Copied";


                setTimeout(
                    () => {

                        copyButton.textContent =
                            original;

                    },
                    1200
                );


            } catch (_error) {

                showError(
                    "The response could not be copied."
                );
            }
        }
    );


    // ========================================================
    // INITIAL STATE
    // ========================================================

    clearResponse();

    promptInput.focus();

})();
</script>

</body>

</html>
"""


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting Jarvis | port=%d | headless=%s",
        CONFIG.port,
        CONFIG.headless,
    )

    logger.info(
        "Browser profile directory: %s",
        CONFIG.user_data_dir,
    )

    logger.info(
        "Production command: "
        "gunicorn --workers 1 --threads 4 "
        "--timeout 180 app:app"
    )

    app.run(
        host="0.0.0.0",
        port=CONFIG.port,
        debug=False,
        threaded=True,
    )
