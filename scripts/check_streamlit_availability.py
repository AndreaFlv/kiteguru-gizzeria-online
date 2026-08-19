"""Verifica e, se necessario, risveglia l'app Streamlit pubblica.

Il controllo usa un browser reale: una risposta HTTP 200 della shell Streamlit
non dimostra che l'app sia sveglia o che il contenuto applicativo sia visibile.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_APP_URL = "https://kiteguru-gizzeria.streamlit.app/"


def health_url(app_url: str) -> str:
    return f"{app_url.rstrip('/')}/~/+/_stcore/health"


def backend_is_healthy(app_url: str, timeout_seconds: float = 20.0) -> bool:
    request = Request(
        health_url(app_url),
        headers={"User-Agent": "KiteGuru-availability-watchdog/1.0"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except OSError:
        return False


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_check(
    app_url: str,
    *,
    browser_executable: str | None,
    timeout_seconds: int,
) -> dict:
    from playwright.sync_api import sync_playwright

    started = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "app_url": app_url,
        "started_at_utc": started.isoformat(timespec="seconds"),
        "woke_app": False,
        "status": "CHECKING",
    }

    with sync_playwright() as playwright:
        launch_args = {"headless": True}
        if browser_executable:
            launch_args["executable_path"] = browser_executable
        browser = playwright.chromium.launch(**launch_args)
        page = browser.new_page(viewport={"width": 390, "height": 844})
        try:
            page.goto(app_url, wait_until="domcontentloaded", timeout=60_000)
            wake_button = page.get_by_role(
                "button",
                name=re.compile(r"Yes, get this app back up!", re.IGNORECASE),
            )
            if wake_button.count() and wake_button.first.is_visible(timeout=5_000):
                wake_button.first.click(timeout=10_000)
                report["woke_app"] = True

            deadline = time.monotonic() + timeout_seconds
            last_title = page.title()
            app_visible = False
            while time.monotonic() < deadline:
                last_title = page.title()
                if backend_is_healthy(app_url):
                    app_frame = page.frame_locator('iframe[title="streamlitApp"]')
                    marker = app_frame.get_by_text("KiteGuru · Gizzeria", exact=False)
                    try:
                        if marker.first.is_visible(timeout=5_000):
                            app_visible = True
                            break
                    except Exception:
                        pass
                page.wait_for_timeout(5_000)

            report.update({
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "page_title": last_title,
                "backend_healthy": backend_is_healthy(app_url),
                "app_content_visible": app_visible,
                "status": "READY" if app_visible else "FAILED",
            })
            if not app_visible:
                raise RuntimeError(
                    "Backend o contenuto KiteGuru non disponibili entro il timeout"
                )
            return report
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-url", default=DEFAULT_APP_URL)
    parser.add_argument("--browser-executable")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("streamlit-availability.json"),
    )
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "app_url": args.app_url,
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "FAILED",
    }
    try:
        report = run_check(
            args.app_url,
            browser_executable=args.browser_executable,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        report.update({
            "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "error_type": type(exc).__name__,
            "error": str(exc)[:1000],
        })
        write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False))
        return 1

    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
