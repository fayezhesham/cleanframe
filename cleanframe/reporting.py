from typing import List
import logging
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


def log_info(msg: str, report: List[str]):
    """Logs an informational message to the logger, report list, and console."""
    logger.info(msg)
    report.append(msg)
    console.log(f"[green]{msg}")


def log_warning(msg: str, report: List[str]):
    """Logs a warning message to the logger, report list, and console."""
    logger.warning(msg)
    report.append(msg)
    console.log(f"[yellow]{msg}")


def log_error(msg: str, report: List[str]):
    """Logs an error message to the logger, report list, and console."""
    logger.error(msg)
    report.append(msg)
    console.log(f"[red]{msg}")


def display_report(report: List[str]):
    """Formats and prints the list of report messages as a console table."""
    table = Table(title="Data Cleaning Report")
    table.add_column("#", style="dim")
    table.add_column("Message", style="bold")

    for i, msg in enumerate(report, start=1):
        table.add_row(str(i), msg)

    console.print(table)


def log_duplicates_found(column: str, count: int, report: List[str]):
    """Logs a warning about duplicate values found in a column."""
    message = f"Found {count} duplicate value(s) in column '{column}'."
    logger.warning(message)
    report.append(message)


def log_duplicates_removed(column: str, removed_count: int, report: List[str]):
    """Logs an informational message about duplicate rows being removed."""
    message = f"Marked {removed_count} duplicate row(s) in column '{column}' for removal, keeping only unique entries."
    logger.info(message)
    report.append(message)
    console.log(f"[yellow]{message}")


def log_regex_invalid(column: str, count: int, report: List[str]):
    """Logs a warning about values failing regex validation."""
    message = f"{count} value(s) in column '{column}' failed regex validation."
    logger.warning(message)
    report.append(message)
    console.log(f"[yellow]{message}")