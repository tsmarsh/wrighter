from datetime import datetime
from pathlib import Path
import glob

import logging

history_dir = Path("chat_history")


def save_message(role, content):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Create subdirectories for year and month
    subdir = history_dir / datetime.now().strftime("%Y/%m")
    subdir.mkdir(parents=True, exist_ok=True)

    # Define the filename
    filename = subdir / f"{timestamp}-{role}.md"

    # Write content to the Markdown file
    with open(filename, "w") as file:
        file.write(f"---\ntimestamp: {timestamp}\nrole: {role}\n---\n\n")
        file.write(content)

    logging.info(f"Saved {role} message to {filename}")


def all_messages():
    return glob.glob(f"{history_dir}/**/*.md", recursive=True)