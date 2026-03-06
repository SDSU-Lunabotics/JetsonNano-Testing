import argparse
import json
import sys
import requests


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a hello world to a backend via HTTP POST")
    parser.add_argument("--url", required=True, help="Backend URL endpoint")
    parser.add_argument("--api-key", default="", help="Optional bearer token")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    args = parser.parse_args()

    payload = {
        "message": "hello world",
        "source": "jetson-nano",
    }

    headers = {
        "Content-Type": "application/json",
    }
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    try:
        resp = requests.post(args.url, json=payload, headers=headers, timeout=args.timeout)
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    print(resp.status_code)
    print(resp.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
