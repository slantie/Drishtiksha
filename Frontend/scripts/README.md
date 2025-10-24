# Markdown Linter & Fixer

A Python utility to automatically detect and fix common markdown linting errors in documentation files.

## Features

- **MD022**: Ensures blank lines around headings
- **MD031**: Ensures blank lines around fenced code blocks
- **MD032**: Ensures blank lines around lists
- **MD040**: Adds language specification to code blocks
- **MD030**: Fixes spacing after list markers

## Usage

### Run the linter

```bash
# From the Frontend directory
python scripts/markdown-linter.py
```

### What it does

The script will:

1. Scan all `.md` files in the `docs/` directory
2. Automatically fix common markdown issues
3. Report which files were modified
4. Preserve HTML tags and special formatting

### Output Example

```text
🔍 Found 4 markdown file(s)

✅ Fixed: docs/backend/README.md
✅ Fixed: docs/frontend/README.md
✓ No changes needed: docs/README.md
✅ Fixed: docs/server/README.md

📊 Summary:
   Total files: 4
   Modified: 3
   Unchanged: 1

✨ Successfully fixed 3 file(s)!
```

## Best Practices

- Run before committing markdown changes
- Can be integrated into pre-commit hooks
- Safe to run multiple times (idempotent)

## Limitations

- Preserves existing HTML tags and special formatting
- Does not modify content, only formatting
- Focuses on structural issues, not content quality
