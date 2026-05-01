import re

with open(r'E:\workspace\codigo_civil_chile.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

def is_starter(line):
    """Check if a line starts a new structural element or paragraph."""
    stripped = line.strip()
    if not stripped:
        return True  # empty line
    if stripped == 'CÓDIGO CIVIL':
        return True
    if re.match(r'^LIBRO\s+', stripped):
        return True
    if re.match(r'^TÍTULO\s+', stripped):
        return True
    if re.match(r'^§\s+\d+', stripped):
        return True
    if re.match(r'^Artículo\s+\S+\.-', stripped):
        return True
    if re.match(r'^Artículo\s+final\.-', stripped):
        return True
    # Numbered items like 1º., 2º., 1., 2., a), b), etc.
    if re.match(r'^\d+[º°]\.', stripped):
        return True
    if re.match(r'^\d+\.', stripped):
        return True
    if re.match(r'^[a-zA-Z]\)', stripped):
        return True
    return False

# Process lines: join continuation lines to previous paragraph
result_lines = []
for line in lines:
    stripped = line.strip()
    if not stripped:
        result_lines.append('')  # keep empty lines as paragraph separators
    elif is_starter(line):
        result_lines.append(stripped)
    else:
        # Continuation line - append to previous non-empty line
        if result_lines and result_lines[-1] != '':
            result_lines[-1] = result_lines[-1] + ' ' + stripped
        else:
            result_lines.append(stripped)

# Clean up excessive blank lines and trailing spaces
output_lines = []
prev_empty = False
for line in result_lines:
    stripped = line.strip()
    if not stripped:
        if not prev_empty:
            output_lines.append('')
            prev_empty = True
    else:
        output_lines.append(stripped)
        prev_empty = False

output_text = '\n'.join(output_lines).strip()

# Save
output_path = r'E:\workspace\codigo_civil_chile.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_text)

word_count = len(output_text.split())
line_count = len(output_lines)
print(f"Saved to: {output_path}")
print(f"Approximate word count: {word_count}")
print(f"Line count: {line_count}")
print(f"Character count: {len(output_text)}")
