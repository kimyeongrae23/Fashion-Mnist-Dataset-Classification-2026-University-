# coding: utf-8
"""Check that Scratch source files do not import deep-learning frameworks."""

import os
import re

FORBIDDEN_IMPORT_PATTERNS = [
    re.compile(r'^\s*import\s+torch(\s|$|\.)'),
    re.compile(r'^\s*from\s+torch(\s|\.)'),
    re.compile(r'^\s*import\s+torchvision(\s|$|\.)'),
    re.compile(r'^\s*from\s+torchvision(\s|\.)'),
    re.compile(r'^\s*import\s+tensorflow(\s|$|\.)'),
    re.compile(r'^\s*from\s+tensorflow(\s|\.)'),
    re.compile(r'^\s*import\s+keras(\s|$|\.)'),
    re.compile(r'^\s*from\s+keras(\s|\.)'),
]

bad = []
for root, dirs, files in os.walk('.'):
    for name in files:
        if not name.endswith('.py'):
            continue
        path = os.path.join(root, name)
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                for pattern in FORBIDDEN_IMPORT_PATTERNS:
                    if pattern.search(line):
                        bad.append((path, line_no, line.strip()))

if bad:
    print('Forbidden deep-learning framework import found:')
    for path, line_no, line in bad:
        print(f'{path}:{line_no}: {line}')
else:
    print('OK: no PyTorch/TensorFlow/Keras import statement found.')
