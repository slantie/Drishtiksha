#!/usr/bin/env python3
"""
Markdown Linter and Fixer
Automatically fixes common markdown linting errors in documentation files.

Fixes:
- MD022: Blank lines around headings
- MD031: Blank lines around fenced code blocks
- MD032: Blank lines around lists
- MD040: Language specification for code blocks
- MD030: Proper spacing after list markers
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


class MarkdownFixer:
    def __init__(self, content: str):
        self.lines = content.split('\n')
        self.fixed_lines = []
        
    def is_heading(self, line: str) -> bool:
        """Check if line is a heading."""
        return bool(re.match(r'^#{1,6}\s+.+', line.strip()))
    
    def is_code_fence(self, line: str) -> bool:
        """Check if line is a code fence (```)."""
        return line.strip().startswith('```')
    
    def is_list_item(self, line: str) -> bool:
        """Check if line is a list item."""
        return bool(re.match(r'^\s*[-*+]\s+.+', line)) or bool(re.match(r'^\s*\d+\.\s+.+', line))
    
    def is_blank(self, line: str) -> bool:
        """Check if line is blank or whitespace only."""
        return len(line.strip()) == 0
    
    def fix_list_markers(self, line: str) -> str:
        """Fix spacing after list markers (MD030)."""
        # Fix multiple spaces after unordered list markers
        line = re.sub(r'^(\s*)[-*+]\s{2,}', r'\1- ', line)
        # Fix multiple spaces after ordered list markers
        line = re.sub(r'^(\s*)(\d+\.)\s{2,}', r'\1\2 ', line)
        return line
    
    def fix_code_fence_language(self, line: str) -> str:
        """Add 'text' language to code fences without language specification (MD040)."""
        if line.strip() == '```':
            return line.replace('```', '```text')
        return line
    
    def process(self) -> str:
        """Process and fix markdown content."""
        i = 0
        in_code_block = False
        in_html_details = False
        
        while i < len(self.lines):
            line = self.lines[i]
            
            # Track HTML details/summary tags
            if '<details>' in line.lower():
                in_html_details = True
            if '</details>' in line.lower():
                in_html_details = False
            
            # Track code blocks
            if self.is_code_fence(line):
                in_code_block = not in_code_block
                
                # Fix code fence without language (MD040)
                if not in_code_block and line.strip() == '```':
                    # Don't add language if it's closing fence
                    pass
                elif in_code_block and line.strip() == '```':
                    line = self.fix_code_fence_language(line)
                
                # Add blank line before code fence if needed (MD031)
                if in_code_block and self.fixed_lines and not self.is_blank(self.fixed_lines[-1]):
                    # Don't add blank if previous line is list item or inside HTML tags
                    if not (self.is_list_item(self.fixed_lines[-1]) or in_html_details):
                        self.fixed_lines.append('')
                
                self.fixed_lines.append(line)
                
                # Add blank line after closing code fence if needed (MD031)
                if not in_code_block and i + 1 < len(self.lines):
                    next_line = self.lines[i + 1]
                    if not self.is_blank(next_line) and not self.is_code_fence(next_line):
                        if not in_html_details:
                            self.fixed_lines.append('')
                
                i += 1
                continue
            
            # Fix list marker spacing (MD030)
            if self.is_list_item(line):
                line = self.fix_list_markers(line)
                
                # Add blank line before list if needed (MD032)
                if self.fixed_lines and not self.is_blank(self.fixed_lines[-1]):
                    prev_line = self.fixed_lines[-1]
                    # Don't add blank if previous is also a list item or heading
                    if not (self.is_list_item(prev_line) or self.is_heading(prev_line)):
                        self.fixed_lines.append('')
                
                self.fixed_lines.append(line)
                
                # Peek ahead to add blank after list if needed (MD032)
                if i + 1 < len(self.lines):
                    next_line = self.lines[i + 1]
                    if not self.is_blank(next_line) and not self.is_list_item(next_line):
                        # Check if there are more list items coming
                        j = i + 1
                        has_more_list = False
                        while j < len(self.lines) and self.is_blank(self.lines[j]):
                            j += 1
                        if j < len(self.lines) and self.is_list_item(self.lines[j]):
                            has_more_list = True
                        
                        if not has_more_list:
                            # This is the end of the list, add blank after
                            if i + 1 < len(self.lines) and not self.is_blank(self.lines[i + 1]):
                                pass  # Will be handled when we encounter non-list item
                
                i += 1
                continue
            
            # Fix headings (MD022)
            if self.is_heading(line):
                # Add blank line before heading if needed
                if self.fixed_lines and not self.is_blank(self.fixed_lines[-1]):
                    self.fixed_lines.append('')
                
                self.fixed_lines.append(line)
                
                # Add blank line after heading if needed
                if i + 1 < len(self.lines):
                    next_line = self.lines[i + 1]
                    if not self.is_blank(next_line):
                        self.fixed_lines.append('')
                
                i += 1
                continue
            
            # Regular line
            self.fixed_lines.append(line)
            i += 1
        
        return '\n'.join(self.fixed_lines)


def fix_markdown_file(file_path: Path) -> Tuple[bool, str]:
    """
    Fix markdown linting issues in a file.
    
    Returns:
        Tuple of (was_modified, message)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        fixer = MarkdownFixer(original_content)
        fixed_content = fixer.process()
        
        if original_content != fixed_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True, f"✅ Fixed: {file_path}"
        else:
            return False, f"✓ No changes needed: {file_path}"
            
    except Exception as e:
        return False, f"❌ Error processing {file_path}: {str(e)}"


def main():
    """Main function to process markdown files."""
    # Get docs directory
    script_dir = Path(__file__).parent
    docs_dir = script_dir.parent / 'docs'
    
    if not docs_dir.exists():
        print(f"❌ Docs directory not found: {docs_dir}")
        sys.exit(1)
    
    # Find all markdown files
    md_files = list(docs_dir.rglob('*.md'))
    
    if not md_files:
        print(f"❌ No markdown files found in {docs_dir}")
        sys.exit(1)
    
    print(f"🔍 Found {len(md_files)} markdown file(s)\n")
    
    modified_count = 0
    results = []
    
    for md_file in sorted(md_files):
        was_modified, message = fix_markdown_file(md_file)
        results.append(message)
        if was_modified:
            modified_count += 1
    
    # Print results
    for result in results:
        print(result)
    
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(md_files)}")
    print(f"   Modified: {modified_count}")
    print(f"   Unchanged: {len(md_files) - modified_count}")
    
    if modified_count > 0:
        print(f"\n✨ Successfully fixed {modified_count} file(s)!")
    else:
        print(f"\n✓ All files are already properly formatted!")


if __name__ == '__main__':
    main()
