#!/usr/bin/env python3
"""Local stdlib HTTP endpoint for a deployable dry-run inference surface."""
import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from ink_ml import InkDQN, InkGAN, InkRNN, tokenize_source

USERS = {}
SESSIONS = {}
RATE_LIMITS = {}
SESSION_SECRET = secrets.token_bytes(32)
PBKDF2_ROUNDS = 210_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"{salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, digest_hex = stored_hash.split("$", 1)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), PBKDF2_ROUNDS).hex()
    return hmac.compare_digest(candidate, digest_hex)


def create_session(email: str) -> str:
    expires = int(time.time()) + 86_400
    payload = f"{email}|{expires}".encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    signature = hmac.new(SESSION_SECRET, encoded.encode(), hashlib.sha256).hexdigest()
    SESSIONS[encoded] = (email, expires, signature)
    return f"{encoded}.{signature}"


def valid_session(cookie: str | None) -> str | None:
    if not cookie or "." not in cookie:
        return None
    encoded, signature = cookie.split(".", 1)
    expected = hmac.new(SESSION_SECRET, encoded.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    session = SESSIONS.get(encoded)
    if not session or session[1] < int(time.time()):
        return None
    return session[0]


def rate_limited(address: str) -> bool:
    now = time.time()
    recent = [stamp for stamp in RATE_LIMITS.get(address, []) if stamp > now - 60]
    recent.append(now)
    RATE_LIMITS[address] = recent
    return len(recent) > 10


class InkHandler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict, cookie: str | None = None):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        if cookie:
            self.send_header("Set-Cookie", f"inkgraph_session={cookie}; HttpOnly; SameSite=Strict; Path=/; Max-Age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if rate_limited(self.client_address[0]):
            self.send_json(429, {"error": "Too many requests; try again later"})
            return
        try:
            payload = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", "0")), 20_000)))
            if self.path in ("/register", "/login"):
                self.handle_auth(payload)
                return
            if self.path != "/transform":
                self.send_json(404, {"error": "Unknown endpoint"})
                return
            source = str(payload["source"])
            if not source or len(source) > 500:
                raise ValueError("source must contain 1-500 characters")
            features = tokenize_source(source)
            rnn, gan, dqn = InkRNN(), InkGAN(), InkDQN()
            response = {"source": source, "controls": rnn.predict(features), "generated_style": gan.generate(features), "recommended_action": dqn.choose()}
            self.send_json(200, response)
        except (KeyError, json.JSONDecodeError, TypeError, ValueError):
            self.send_json(400, {"error": "Expected valid JSON with a source field"})

    def handle_auth(self, payload: dict):
        email = str(payload.get("email", "")).strip().lower()
        password = str(payload.get("password", ""))
        if not EMAIL_PATTERN.match(email) or len(email) > 254 or not 12 <= len(password) <= 128:
            self.send_json(400, {"error": "Use a valid email and a 12-128 character password"})
            return
        if self.path == "/register":
            if email in USERS:
                self.send_json(409, {"error": "Unable to create account"})
                return
            USERS[email] = hash_password(password)
            self.send_json(201, {"message": "Account created"}, create_session(email))
            return
        if email not in USERS or not verify_password(password, USERS[email]):
            self.send_json(401, {"error": "Invalid email or password"})
            return
        self.send_json(200, {"message": "Signed in"}, create_session(email))

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    print("Inkgraph dry-run service listening on http://127.0.0.1:8765")
    HTTPServer(("127.0.0.1", 8765), InkHandler).serve_forever()