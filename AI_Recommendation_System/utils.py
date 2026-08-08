"""Reusable terminal helpers for the AI Recommendation System."""

from __future__ import annotations

import sys
import time
from typing import Any


class Color:
    """ANSI colors used by the terminal interface."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def color_text(text: str, color: str) -> str:
    """Return colored terminal text."""
    return f"{color}{text}{Color.RESET}"


def print_banner() -> None:
    """Display the professional ASCII project banner."""
    banner = r"""
================================================
        AI RECOMMENDATION SYSTEM
================================================
       Content-Based Filtering with TF-IDF
             and Cosine Similarity
================================================
"""
    print(color_text(banner, Color.CYAN))


def print_menu() -> None:
    """Display the application menu."""
    print(color_text("\n+----------------------------------------------+", Color.BLUE))
    print(color_text("|                  MAIN MENU                   |", Color.BLUE))
    print(color_text("+----------------------------------------------+", Color.BLUE))
    print("| 1. View Dataset                              |")
    print("| 2. Dataset Statistics                        |")
    print("| 3. Search Item                               |")
    print("| 4. Recommend Items                           |")
    print("| 5. View Similarity Score                     |")
    print("| 6. Show Recommendation Pipeline              |")
    print("| 7. Exit                                      |")
    print(color_text("+----------------------------------------------+", Color.BLUE))


def loading_step(message: str, action: Any) -> Any:
    """Run one setup action with a small loading animation."""
    print(f"{Color.YELLOW}{message}...{Color.RESET}", end="", flush=True)
    frames = ["|", "/", "-", "\\"]
    for frame in frames:
        print(f"\r{Color.YELLOW}{message}... {frame}{Color.RESET}",
              end="", flush=True)
        time.sleep(0.08)

    result = action()
    print(f"\r{Color.GREEN}{message}... Done{Color.RESET}")
    return result


def print_success(message: str) -> None:
    print(color_text(message, Color.GREEN))


def print_error(message: str) -> None:
    print(color_text(f"Error: {message}", Color.RED))


def print_warning(message: str) -> None:
    print(color_text(f"Warning: {message}", Color.YELLOW))


def print_info(message: str) -> None:
    print(color_text(message, Color.CYAN))


def prompt_non_empty(message: str) -> str:
    """Ask for non-empty user input."""
    while True:
        value = input(color_text(message, Color.WHITE)).strip()
        if value:
            return value
        print_error("Input cannot be empty. Please try again.")


def prompt_integer(
    message: str,
    minimum: int,
    maximum: int,
    default: int | None = None,
) -> int:
    """Ask for an integer within a valid range."""
    while True:
        value = input(color_text(message, Color.WHITE)).strip()
        if not value and default is not None:
            return default

        try:
            number = int(value)
            if minimum <= number <= maximum:
                return number
            print_error(f"Enter a number from {minimum} to {maximum}.")
        except ValueError:
            print_error("Enter a valid number.")


def ask_continue() -> bool:
    """Ask whether the user wants to continue using the application."""
    while True:
        choice = input(
            color_text("\nDo you want to continue? (y/n): ", Color.WHITE)
        ).strip().lower()

        if choice in {"y", "yes", ""}:
            return True
        if choice in {"n", "no"}:
            return False
        print_error("Please enter y or n.")


def truncate_text(value: Any, max_width: int) -> str:
    """Shorten long table values so the terminal layout remains clean."""
    text = str(value)
    if len(text) <= max_width:
        return text
    return text[: max_width - 3] + "..."


def format_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    """Create a clean ASCII table from a list of dictionaries."""
    if not rows:
        return "No records found."

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {header: truncate_text(row.get(header, ""), 28)
             for header in headers}
        )

    widths = {
        header: max(
            len(header),
            max(len(str(row[header])) for row in normalized_rows),
        )
        for header in headers
    }

    border = "+"
    for header in headers:
        border += "-" * (widths[header] + 2) + "+"

    header_line = "|"
    for header in headers:
        header_line += f" {header.ljust(widths[header])} |"

    lines = [border, header_line, border]
    for row in normalized_rows:
        line = "|"
        for header in headers:
            line += f" {str(row[header]).ljust(widths[header])} |"
        lines.append(line)
    lines.append(border)
    return "\n".join(lines)


def dataframe_to_rows(dataframe: Any) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to table rows without importing pandas here."""
    return dataframe.to_dict(orient="records")


def print_section(title: str) -> None:
    """Print a section heading."""
    print(color_text(f"\n{'=' * 48}", Color.MAGENTA))
    print(color_text(title.center(48), Color.MAGENTA))
    print(color_text(f"{'=' * 48}", Color.MAGENTA))


def exit_application() -> None:
    """Close the application with a friendly message."""
    print_success("\nThank you for using the AI Recommendation System.")
    print_success("Application closed successfully.")
    sys.exit(0)
