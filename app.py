import os
import sys
import time
import logging
import threading
import atexit
from typing import Optional, Tuple, Dict, Any, List

from flask import Flask, request, jsonify, render_template_string
from playwright.sync_api import sync_playwright, Playwright, BrowserContext, Page, Locator, Error as PlaywrightError


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Central configuration class populated from environment variables."""
    PORT: int = int(os.environ.get("PORT", "10000"))
    HEADLESS: bool = os.environ.get("HEADLESS", "true").lower() in ("true", "1", "yes")
    CHATGPT_URL: str = os.environ.get("CHATGPT_URL", "https://chatgpt.com/")
    PLAYWRIGHT_USER_DATA_DIR: str = os.environ.get("PLAYWRIGHT_USER_DATA_DIR", "./playwright_data")
    CHAT_TIMEOUT_SECONDS: int = int(os.environ.get("CHAT_TIMEOUT_SECONDS", "120"))
    BROWSER_LOCK_TIMEOUT_SECONDS: int = int(os.environ.get("BROWSER_LOCK_TIMEOUT_SECONDS", "180"))
    MAX_PROMPT_LENGTH: int = int(os.environ.get("MAX_PROMPT_LENGTH", "10000"))
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()


# Configure Logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("chatgpt-gateway")


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class AutomationError(Exception):
    """Base exception for web automation failures."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(AutomationError):
    """Raised when the browser session is unauthenticated."""
    def __init__(self, message: str = "ChatGPT session is not authenticated. Authenticate the persistent browser profile first."):
        super().__init__(message, status_code=401)


class PromptInputNotFoundError(AutomationError):
    """Raised when the prompt textarea cannot be located."""
    def __init__(self, message: str = "Unable to locate the prompt input field on ChatGPT. DOM structure may have changed."):
        super().__init__(message, status_code=503)


class SendButtonNotFoundError(AutomationError):
    """Raised when the submission button/mechanism cannot be executed."""
    def __init__(self, message: str = "Unable to locate or trigger the send button on ChatGPT."):
        super().__init__(message, status_code=503)


class ResponseTimeoutError(AutomationError):
    """Raised when assistant response generation times out."""
    def __init__(self, message: str = "Timed out waiting for ChatGPT to generate a complete response."):
        super().__init__(message, status_code=504)


class BrowserBusyError(AutomationError):
    """Raised when another request holds the browser lock for too long."""
    def __init__(self, message: str = "Browser session is currently busy processing another request. Please try again."):
        super().__init__(message, status_code=409)


# ==============================================================================
# DOM SELECTORS STRATEGY
# ==============================================================================

PROMPT_SELECTORS: List[str] = [
    "#prompt-textarea",
    "textarea[data-id='root']",
    "div[id='prompt-textarea']",
    "textarea[placeholder*='Message']",
    "p[data-placeholder]",
    "textarea",
]

SEND_BUTTON_SELECTORS: List[str] = [
    "button[data-testid='aria-send-button']",
    "button[data-testid='send-button']",
    "button[aria-label='Send prompt']",
    "button[aria-label='Send message']",
    "button[aria-label='Submit']",
]

STOP_BUTTON_SELECTORS: List[str] = [
    "button[data-testid='stop-button']",
    "button[aria-label='Stop streaming']",
    "button[aria-label='Stop generating']",
    "button[aria-label='Stop']",
]

ASSISTANT_MESSAGE_SELECTORS: List[str] = [
    "[data-message-author-role='assistant']",
    "div[data-is-streaming]",
    ".agent-turn",
    "div.markdown",
]

LOGIN_INDICATORS: List[str] = [
    "button[data-testid='login-button']",
    "a[href*='/auth/login']",
    "button:has-text('Log in')",
    "button:has-text('Sign up')",
]


# ==============================================================================
# PLAYWRIGHT BROWSER MANAGER
# ==============================================================================

class BrowserManager:
    """
    Thread-safe Browser Manager managing persistent Playwright Chromium instance.
    Serializes all browser interactions using a threading lock.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.playwright: Optional[Playwright] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize persistent browser context lazily."""
        if self._is_initialized and self.page and not self.page.is_closed():
            return

        logger.info("Initializing Playwright Persistent Context...")
        os.makedirs(Config.PLAYWRIGHT_USER_DATA_DIR, exist_ok=True)

        try:
            self.playwright = sync_playwright().start()
            
            # Note: Standard launch arguments required for container environments like Render.
            # Security bypass arguments are intentionally omitted to align with standard platform terms.
            launch_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
            ]

            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=Config.PLAYWRIGHT_USER_DATA_DIR,
                headless=Config.HEADLESS,
                args=launch_args,
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )

            pages = self.context.pages
            self.page = pages[0] if pages else self.context.new_page()
            self.page.set_default_timeout(15000)

            logger.info("Navigating to ChatGPT URL: %s", Config.CHATGPT_URL)
            self.page.goto(Config.CHATGPT_URL, wait_until="domcontentloaded", timeout=30000)
            self._is_initialized = True
            logger.info("Browser persistent session successfully initialized.")

        except Exception as e:
            logger.error("Failed to initialize browser session: %s", str(e), exc_info=True)
            self.cleanup()
            raise AutomationError(f"Failed to initialize browser automation engine: {str(e)}", status_code=503)

    def is_healthy(self) -> bool:
        """Lightweight health check for browser state."""
        return bool(self._is_initialized and self.page and not self.page.is_closed())

    def verify_authentication(self) -> None:
        """Check whether the page exhibits unauthenticated login prompts."""
        if not self.page:
            raise AutomationError("Page instance not available", status_code=503)

        for selector in LOGIN_INDICATORS:
            try:
                elem = self.page.locator(selector).first
                if elem.is_visible(timeout=1000):
                    logger.warning("Detected unauthenticated login indicator: %s", selector)
                    raise AuthenticationError()
            except PlaywrightError:
                continue

    def find_prompt_input(self) -> Locator:
        """Locate visible and editable prompt textarea using fallback strategies."""
        if not self.page:
            raise AutomationError("Page instance not available", status_code=503)

        for selector in PROMPT_SELECTORS:
            try:
                locator = self.page.locator(selector).first
                if locator.is_visible(timeout=1500):
                    return locator
            except PlaywrightError:
                continue

        logger.error("No prompt input selector matched.")
        raise PromptInputNotFoundError()

    def find_send_button(self) -> Optional[Locator]:
        """Locate send button if available."""
        if not self.page:
            return None

        for selector in SEND_BUTTON_SELECTORS:
            try:
                locator = self.page.locator(selector).first
                if locator.is_visible(timeout=1000):
                    return locator
            except PlaywrightError:
                continue

        return None

    def send_prompt(self, prompt: str) -> int:
        """
        Types prompt into input area and submits message.
        Returns initial count of assistant messages prior to submission.
        """
        self.verify_authentication()
        
        # Capture current assistant message count
        initial_assistant_count = self.get_assistant_message_count()

        input_locator = self.find_prompt_input()
        input_locator.focus()
        input_locator.fill(prompt)
        time.sleep(0.2)

        send_button = self.find_send_button()
        if send_button and send_button.is_enabled():
            send_button.click()
            logger.info("Submitted prompt via send button click.")
        else:
            # Fallback to Keyboard Enter
            logger.info("Send button not immediately clickable; submitting via Enter key.")
            input_locator.press("Enter")

        return initial_assistant_count

    def get_assistant_message_count(self) -> int:
        """Count current assistant responses rendered in DOM."""
        if not self.page:
            return 0

        for selector in ASSISTANT_MESSAGE_SELECTORS:
            try:
                count = self.page.locator(selector).count()
                if count > 0:
                    return count
            except PlaywrightError:
                continue
        return 0

    def wait_for_response_completion(self, initial_count: int) -> None:
        """
        Polls DOM to detect response streaming start and stability/completion.
        """
        if not self.page:
            raise AutomationError("Page not available", status_code=503)

        start_time = time.time()
        timeout = Config.CHAT_TIMEOUT_SECONDS

        logger.info("Waiting for assistant streaming response...")

        # Step 1: Wait for response generation to begin (new message created or stop button visible)
        generation_started = False
        while time.time() - start_time < 15:
            current_count = self.get_assistant_message_count()
            has_stop_button = any(
                self.page.locator(sel).first.is_visible(timeout=200)
                for sel in STOP_BUTTON_SELECTORS
                if self._safe_is_visible(sel)
            )

            if current_count > initial_count or has_stop_button:
                generation_started = True
                break
            time.sleep(0.5)

        if not generation_started:
            logger.warning("Did not detect explicit streaming start signal within 15 seconds; checking output directly.")

        # Step 2: Wait until generation stops (stop button disappears & content stabilizes)
        last_text = ""
        stable_count = 0

        while time.time() - start_time < timeout:
            has_stop_button = any(
                self._safe_is_visible(sel) for sel in STOP_BUTTON_SELECTORS
            )

            current_text = self.extract_latest_assistant_response()

            if not has_stop_button and len(current_text) > 0:
                if current_text == last_text:
                    stable_count += 1
                    if stable_count >= 2:  # Text unchanged for ~1 sec after stop button vanishes
                        logger.info("Response text stabilized successfully.")
                        return
                else:
                    stable_count = 0
                    last_text = current_text

            time.sleep(0.5)

        if len(last_text) > 0:
            logger.warning("Response generation timed out but partial content was retrieved.")
            return

        raise ResponseTimeoutError()

    def _safe_is_visible(self, selector: str) -> bool:
        """Helper to safely check visibility without throwing on missing elements."""
        try:
            return self.page.locator(selector).first.is_visible(timeout=100) if self.page else False
        except PlaywrightError:
            return False

    def extract_latest_assistant_response(self) -> str:
        """Extract and clean visible text from the latest assistant message element."""
        if not self.page:
            return ""

        for selector in ASSISTANT_MESSAGE_SELECTORS:
            try:
                locators = self.page.locator(selector)
                count = locators.count()
                if count > 0:
                    latest_element = locators.nth(count - 1)
                    raw_text = latest_element.inner_text()
                    cleaned_text = raw_text.strip()
                    if cleaned_text:
                        return cleaned_text
            except PlaywrightError:
                continue

        return ""

    def process_prompt(self, prompt: str) -> str:
        """
        Thread-safe entry point to execute prompt interaction workflow.
        Acquires browser lock, initializes session, handles submission, and returns reply.
        """
        acquired = self.lock.acquire(timeout=Config.BROWSER_LOCK_TIMEOUT_SECONDS)
        if not acquired:
            raise BrowserBusyError()

        try:
            # Ensure initialization / recovery
            if not self.is_healthy():
                self.initialize()

            # Ensure page is at ChatGPT
            if self.page and Config.CHATGPT_URL not in self.page.url:
                logger.info("Navigating back to main ChatGPT URL...")
                self.page.goto(Config.CHATGPT_URL, wait_until="domcontentloaded", timeout=20000)

            initial_count = self.send_prompt(prompt)
            self.wait_for_response_completion(initial_count)
            reply = self.extract_latest_assistant_response()

            if not reply:
                raise AutomationError("Assistant response was empty or could not be extracted.", status_code=500)

            return reply

        except AutomationError:
            raise
        except Exception as e:
            logger.error("Unexpected error during Playwright interaction: %s", str(e), exc_info=True)
            self.cleanup()  # Force re-initialization on next request upon crash
            raise AutomationError(f"Automation engine encountered an internal failure: {str(e)}", status_code=500)
        finally:
            self.lock.release()

    def cleanup(self) -> None:
        """Close browser context and Playwright cleanly."""
        logger.info("Cleaning up Playwright resources...")
        try:
            if self.context:
                self.context.close()
            if self.playwright:
                self.playwright.stop()
        except Exception as e:
            logger.warning("Error during Playwright cleanup: %s", str(e))
        finally:
            self.context = None
            self.page = None
            self.playwright = None
            self._is_initialized = False


# Instantiate Singleton BrowserManager
browser_manager = BrowserManager()

# Ensure browser cleanup on process exit
atexit.register(browser_manager.cleanup)


# ==============================================================================
# FLASK APPLICATION SETUP
# ==============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max body


# ==============================================================================
# SECURITY HEADERS MIDDLWARE
# ==============================================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# ==============================================================================
# FRONTEND HTML TEMPLATE
# ==============================================================================

INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="h-full bg-slate-900 text-slate-100">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT Web Automation Gateway</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .pre-wrap { white-space: pre-wrap; word-break: break-word; }
    </style>
</head>
<body class="h-full flex flex-col font-sans antialiased">
    <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-10">
        <div class="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="h-3 w-3 rounded-full bg-emerald-500 animate-pulse"></div>
                <h1 class="text-lg font-bold tracking-tight text-white">ChatGPT Web Automation Gateway</h1>
            </div>
            <span class="text-xs font-medium px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700">
                Production-Ready
            </span>
        </div>
    </header>

    <main class="flex-1 max-w-5xl w-full mx-auto p-4 flex flex-col gap-6">
        <!-- Input Form -->
        <section class="bg-slate-800/50 rounded-xl border border-slate-700/60 p-5 shadow-lg">
            <form id="chatForm" class="flex flex-col gap-3">
                <div class="flex justify-between items-center text-xs text-slate-400 font-medium">
                    <label for="prompt" class="uppercase tracking-wider">Prompt Input</label>
                    <span id="charCounter">0 / {{ max_prompt_length }}</span>
                </div>
                <textarea 
                    id="prompt" 
                    name="prompt" 
                    rows="5" 
                    maxlength="{{ max_prompt_length }}"
                    placeholder="Type your message here... (Press Ctrl+Enter or click Send)" 
                    class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3.5 text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition text-sm resize-y"
                    required
                ></textarea>

                <div class="flex justify-between items-center pt-2">
                    <div class="text-xs text-slate-500">
                        Shortcut: <kbd class="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">Ctrl</kbd> + <kbd class="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300">Enter</kbd>
                    </div>
                    <button 
                        type="submit" 
                        id="submitBtn"
                        class="inline-flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm px-5 py-2.5 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-emerald-400 disabled:opacity-50 disabled:cursor-not-allowed shadow-md"
                    >
                        <span id="btnText">Send Prompt</span>
                        <svg id="btnSpinner" class="hidden animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                    </button>
                </div>
            </form>
        </section>

        <!-- Error Banner -->
        <div id="errorBanner" class="hidden bg-rose-950/80 border border-rose-800 text-rose-200 p-4 rounded-xl text-sm flex items-start gap-3">
            <svg class="h-5 w-5 text-rose-400 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d=" " />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div id="errorMessage" class="flex-1 font-mono text-xs"></div>
        </div>

        <!-- Output Display -->
        <section class="flex-1 bg-slate-800/30 rounded-xl border border-slate-700/40 p-5 flex flex-col min-h-[250px]">
            <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Assistant Response</h2>
            <div id="responseContainer" class="flex-1 bg-slate-950/60 rounded-lg border border-slate-800 p-4 font-mono text-xs text-slate-200 overflow-y-auto pre-wrap">
                <span class="text-slate-600 italic">No response generated yet. Send a prompt to receive output.</span>
            </div>
        </section>
    </main>

    <script>
        const chatForm = document.getElementById('chatForm');
        const promptInput = document.getElementById('prompt');
        const charCounter = document.getElementById('charCounter');
        const submitBtn = document.getElementById('submitBtn');
        const btnText = document.getElementById('btnText');
        const btnSpinner = document.getElementById('btnSpinner');
        const errorBanner = document.getElementById('errorBanner');
        const errorMessage = document.getElementById('errorMessage');
        const responseContainer = document.getElementById('responseContainer');

        const MAX_LENGTH = {{ max_prompt_length }};

        // Character counter handler
        promptInput.addEventListener('input', () => {
            const len = promptInput.value.length;
            charCounter.textContent = `${len} / ${MAX_LENGTH}`;
        });

        // Ctrl+Enter keyboard submission
        promptInput.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'Enter') {
                e.preventDefault();
                chatForm.requestSubmit();
            }
        });

        // Form submission handler
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const promptText = promptInput.value.trim();

            if (!promptText) return;

            // UI State: Loading
            submitBtn.disabled = true;
            btnSpinner.classList.remove('hidden');
            btnText.textContent = 'Processing...';
            errorBanner.classList.add('hidden');
            responseContainer.innerHTML = '<span class="text-amber-400/80 animate-pulse">Waiting for ChatGPT response... This may take up to 2 minutes depending on response size.</span>';

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: promptText })
                });

                const data = await res.json();

                if (!res.ok) {
                    throw new Error(data.error || `HTTP ${res.status}: Server Error`);
                }

                // Render Response
                responseContainer.textContent = data.reply;

            } catch (err) {
                errorMessage.textContent = err.message || 'An unexpected network error occurred.';
                errorBanner.classList.remove('hidden');
                responseContainer.innerHTML = '<span class="text-rose-400 italic">Request failed. Check error message above.</span>';
            } finally {
                // Restore Button State
                submitBtn.disabled = false;
                btnSpinner.classList.add('hidden');
                btnText.textContent = 'Send Prompt';
            }
        });
    </script>
</body>
</html>
"""


# ==============================================================================
# ROUTES
# ==============================================================================

@app.route("/", methods=["GET"])
def index():
    """Render application HTML frontend."""
    return render_template_string(INDEX_HTML, max_prompt_length=Config.MAX_PROMPT_LENGTH)


@app.route("/health", methods=["GET"])
def health():
    """Return lightweight health check indicator without triggering web automation."""
    return jsonify({
        "status": "ok",
        "service": "chatgpt-web-automation",
        "browser_initialized": browser_manager.is_healthy()
    }), 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """API Endpoint to send prompts to ChatGPT via Playwright automation."""
    if not request.is_json:
        return jsonify({"error": "Request content type must be application/json"}), 400

    data = request.get_json(silent=True)
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field in request payload."}), 400

    prompt = data.get("prompt")

    if not isinstance(prompt, str):
        return jsonify({"error": "Prompt must be a string."}), 400

    prompt = prompt.strip()
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty or whitespace only."}), 400

    if len(prompt) > Config.MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Prompt exceeds maximum character length of {Config.MAX_PROMPT_LENGTH}."}), 400

    try:
        reply = browser_manager.process_prompt(prompt)
        return jsonify({"reply": reply}), 200

    except AutomationError as ae:
        return jsonify({"error": ae.message}), ae.status_code
    except Exception as e:
        logger.error("Unhandled API Error: %s", str(e), exc_info=True)
        return jsonify({"error": "An internal server error occurred while processing your request."}), 500


# ==============================================================================
# ENTRY POINT
# ==============================================================================

"""
RENDER DEPLOYMENT & GUNICORN ADVISORY:
- Build Command: 
    pip install flask playwright gunicorn && playwright install chromium
- Start Command: 
    gunicorn --workers 1 --threads 4 --timeout 300 app:app

Note: Set worker count to 1 (`--workers 1`) on Render. Playwright persistent browser context
maintains local state on disk and should reside within a single process to avoid locks on the profile directory.
"""

if __name__ == "__main__":
    logger.info("Starting Flask development server on host 0.0.0.0 port %d", Config.PORT)
    app.run(host="0.0.0.0", port=Config.PORT, debug=False)
