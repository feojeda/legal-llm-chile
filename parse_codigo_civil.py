from bs4 import BeautifulSoup
import html
import re

# Read the HTML file
with open(r'C:\Users\feoje\.local\share\opencode\tool-output\tool_de3af3012001rRaQglkItPLKYm', 'r', encoding='utf-8') as f:
    html_content = f.read()

# Parse with BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the PRE tag that contains the code content
pre = soup.find('pre')
if not pre:
    pre = soup.find('PRE')

if not pre:
    raise ValueError("No PRE tag found")

# Get all text content, but we need to preserve some structure
# Convert to string and manually process to keep articles and structure
pre_html = str(pre)

# Remove the opening and closing PRE tags
pre_html = re.sub(r'^<PRE>\s*', '', pre_html, flags=re.IGNORECASE)
pre_html = re.sub(r'\s*</PRE>\s*$', '', pre_html, flags=re.IGNORECASE)

# Remove version/modification line
pre_html = re.sub(r'<b[^>]*class=["\']version["\'][^>]*>.*?</b>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Remove anchor tags entirely (keep content if any, but anchors are empty)
pre_html = re.sub(r'<a\s+[^>]*>\s*</a>', '', pre_html, flags=re.IGNORECASE)
pre_html = re.sub(r'<a\s+[^>]*>\s*(.*?)\s*</a>', r'\1', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Remove span tags but keep their content
pre_html = re.sub(r'<span\s+[^>]*>(.*?)</span>', r'\1', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Remove BR tags, replace with newlines
pre_html = re.sub(r'<BR\s*/?>', '\n', pre_html, flags=re.IGNORECASE)
pre_html = re.sub(r'<br\s*/?>', '\n', pre_html, flags=re.IGNORECASE)

# Remove table, tr, td tags (navigation like bis/ter links)
pre_html = re.sub(r'<div[^>]*class=["\']bis["\'][^>]*>.*?</div>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)
pre_html = re.sub(r'<table[^>]*>.*?</table>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)
pre_html = re.sub(r'<tr[^>]*>.*?</tr>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)
pre_html = re.sub(r'<td[^>]*>.*?</td>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Remove other div tags
pre_html = re.sub(r'<div[^>]*>.*?</div>', '', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Remove b tags for articles but keep the text
# The article tags look like: <b class='articulo'>Artículo 1.-</b>
pre_html = re.sub(r'<b\s+class=["\']articulo["\']\s*>(.*?)</b>', r'\1', pre_html, flags=re.DOTALL | re.IGNORECASE)

# For other b tags, keep content (like 1º., 2º., etc.)
pre_html = re.sub(r'<b\s*>(.*?)</b>', r'\1', pre_html, flags=re.DOTALL | re.IGNORECASE)

# Decode HTML entities
pre_html = html.unescape(pre_html)

# Replace &nbsp; and other spaces (html.unescape handles &nbsp;)
# But there might be actual unicode non-breaking spaces
pre_html = pre_html.replace('\xa0', ' ')

# Clean up lines: strip trailing/leading spaces, but preserve indentation that is meaningful
lines = pre_html.split('\n')
cleaned_lines = []
for line in lines:
    # Remove leading/trailing whitespace but keep internal structure
    stripped = line.strip()
    if stripped:
        cleaned_lines.append(stripped)
    else:
        cleaned_lines.append('')

# Now join lines and clean up excessive blank lines
result = '\n'.join(cleaned_lines)

# Replace multiple consecutive blank lines with a single blank line
result = re.sub(r'\n{3,}', '\n\n', result)

# Also clean up spaces before paragraphs that were indented with spaces in PRE
# In the original, article body paragraphs are indented with 5 spaces
# We should preserve that as just the text

# Final cleanup: ensure it starts with CÓDIGO CIVIL
result = result.strip()

# If it doesn't start with CÓDIGO CIVIL, find it and trim before
idx = result.find('CÓDIGO CIVIL')
if idx != -1:
    result = result[idx:]

# Remove any trailing navigation/footer after the last article
# We can look for common patterns at the end
# But for now, let's just save what we have

# Save the output
output_path = r'E:\workspace\codigo_civil_chile.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(result)

# Count words
word_count = len(result.split())
print(f"Saved to: {output_path}")
print(f"Approximate word count: {word_count}")
print(f"Character count: {len(result)}")
