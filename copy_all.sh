#!/bin/bash
# copy_all_py.sh
# Concatenates all Python files in the repo and copies them to your clipboard.

# Combine all .py files (excluding venv/__pycache__)
CONTENT=$(find . -type f -name "*.py" \
  -not -path "*/venv/*" -not -path "*/__pycache__/*" \
  -exec bash -c 'echo "=== {} ==="; cat "{}"; echo; echo' \;)

# Copy to clipboard (auto-detect OS)
if command -v pbcopy &> /dev/null; then
    echo "$CONTENT" | pbcopy
    echo "✅ Copied all Python files to clipboard (macOS)"
elif command -v xclip &> /dev/null; then
    echo "$CONTENT" | xclip -selection clipboard
    echo "✅ Copied all Python files to clipboard (Linux + xclip)"
elif command -v wl-copy &> /dev/null; then
    echo "$CONTENT" | wl-copy
    echo "✅ Copied all Python files to clipboard (Linux + wl-copy)"
elif command -v clip &> /dev/null; then
    echo "$CONTENT" | clip
    echo "✅ Copied all Python files to clipboard (Windows WSL)"
else
    echo "⚠️ No clipboard tool found. Install pbcopy, xclip, wl-copy, or clip."
    exit 1
fi
