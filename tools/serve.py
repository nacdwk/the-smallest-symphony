#!/usr/bin/env python3
"""Dev server: serves the project, wrapping page.html in the same skeleton the Artifact host adds."""
import http.server, os, sys, socketserver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765

SKELETON = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}body{margin:0;font:14px system-ui,sans-serif;background:#fafaf7}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
%s
</body></html>"""

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]
        if path == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        if path in ("/", "/index.html"):
            page = "dist/page.html" if os.environ.get("DIST") else "page.html"
            with open(os.path.join(ROOT, page), "r", encoding="utf-8") as f:
                body = SKELETON.replace("%s", f.read(), 1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()
    def log_message(self, *a):
        pass

class TS(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

TS(("127.0.0.1", PORT), H).serve_forever()
