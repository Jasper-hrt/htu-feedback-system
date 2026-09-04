import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
with open('rendered.html', encoding='utf-8') as f:
    content = f.read()
# Find first showResponseModal
idx = content.find('showResponseModal(54')
print('Found at idx', idx)
print('Around:')
print(repr(content[idx-50:idx+300]))