#!/bin/bash
# Bundle src/ into one inline module and write dist/page.html (the file that gets published).
# three and its addons stay external: the importmap in page.html loads them from jsDelivr.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
mkdir -p dist
npx --yes esbuild@0.24.0 src/main.js --bundle --format=esm --target=es2022 --minify \
  --external:three --external:'three/addons/*' --legal-comments=none --log-level=warning \
  --outfile=dist/bundle.js
python3 - <<'EOF'
import re
page = open('page.html', encoding='utf-8').read()
js = open('dist/bundle.js', encoding='utf-8').read().replace('</script', '<\\/script')
tag = '<script type="module" src="./src/main.js"></script>'
assert tag in page, 'module tag not found in page.html'
out = page.replace(tag, '<script type="module">\n' + js + '\n</script>')
open('dist/page.html', 'w', encoding='utf-8').write(out)
print(f"dist/page.html: {len(out.encode('utf-8'))/1024:.1f} KB  (Claude artifact body; the host adds the document skeleton)")
# Standalone single-file version: the same page wrapped in a full HTML document.
# Open it directly in a browser or host it anywhere static (GitHub Pages, Netlify, S3).
doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
       '<style>[hidden]{display:none!important}body{margin:0}</style>\n</head>\n<body>\n'
       + out + '\n</body>\n</html>\n')
open('dist/index.html', 'w', encoding='utf-8').write(doc)
print(f"dist/index.html: {len(doc.encode('utf-8'))/1024:.1f} KB  (standalone)")
EOF
