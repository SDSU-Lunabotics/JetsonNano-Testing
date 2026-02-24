from http.server import BaseHTTPRequestHandler, HTTPServer
import json

HOST = "0.0.0.0"
PORT = 8000


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {"raw": body}

        print("Received POST:")
        print(f"Path: {self.path}")
        print(f"Headers: {dict(self.headers)}")
        print(f"Body: {data}")

        resp = {"status": "ok", "received": data}
        resp_bytes = json.dumps(resp).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)

    def log_message(self, format, *args):
        return  # silence default logging


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Listening on http://{HOST}:{PORT}")
    server.serve_forever()
