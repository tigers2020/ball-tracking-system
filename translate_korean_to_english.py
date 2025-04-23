#!/usr/bin/env python3
"""Translate Korean text in project files to English.
Requires: pip install deep-translator
"""

import os
import re
import argparse
from deep_translator import GoogleTranslator

# Regex pattern to match Korean characters
KOREAN_REGEX = re.compile(r'[go-Yes]+')

# File extensions to process
SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.html', '.json', '.yaml', '.yml'}

# Directories to skip during traversal
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', 'venv', '.venv'}


def translate_text(text: str, translator: GoogleTranslator) -> str:
    """Translate all Korean sequences in the given text to English."""
    def replace_korean(match):
        korean_text = match.group(0)
        try:
            english_text = translator.translate(korean_text)
            return english_text
        except Exception as error:
            print(f"[Warning] Translation failed for '{korean_text}': {error}")
            return korean_text

    return KOREAN_REGEX.sub(replace_korean, text)


def process_file(file_path: str, translator: GoogleTranslator) -> None:
    """Read file content, translate Korean, and overwrite if changes are made."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except (UnicodeDecodeError, IOError) as error:
        print(f"[Skip] Could not read file {file_path}: {error}")
        return

    translated_content = translate_text(content, translator)
    if translated_content != content:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(translated_content)
            print(f"[Translated] {file_path}")
        except IOError as error:
            print(f"[Error] Could not write file {file_path}: {error}")


def translate_project(root_dir: str) -> None:
    """Walk through project directory and translate supported files."""
    translator = GoogleTranslator(source='ko', target='en')
    for dirpath, dirs, files in os.walk(root_dir):
        # Skip version control, cache, and virtual env dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                process_file(os.path.join(dirpath, filename), translator)


def main():
    parser = argparse.ArgumentParser(
        description="Translate Korean text in project files to English."
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default=os.getcwd(),
        help="Root directory of the project to translate (default: current directory)."
    )
    args = parser.parse_args()
    print(f"Starting translation in: {args.root_dir}")
    translate_project(args.root_dir)
    print("Translation completed.")


if __name__ == '__main__':
    main() 