import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.environ.get("VLLM_HEALTH_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = float(os.environ.get("VLLM_HEALTH_TIMEOUT", "5"))


def check(url: str, token: str | None = None) -> tuple[bool, str]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            status = response.getcode()
            if status < 400:
                return True, f"{url} returned HTTP {status}"
            return False, f"{url} returned HTTP {status}"
    except urllib.error.HTTPError as error:
        return False, f"{url} returned HTTP {error.code}"
    except Exception as error:
        return False, f"{url} failed: {error}"


def main() -> int:
    tokens = [
        os.environ.get("RAGAS_LLM_API_KEY"),
        os.environ.get("GENERATOR_API_KEY"),
        os.environ.get("VLLM_API_KEY"),
        "local-vllm-key",
    ]
    tokens = list(dict.fromkeys(token for token in tokens if token))

    attempts: list[tuple[str, str | None]] = [(f"{BASE_URL}/health", None)]
    attempts.extend((f"{BASE_URL}/v1/models", token) for token in tokens)
    attempts.append((f"{BASE_URL}/v1/models", None))

    errors: list[str] = []
    for url, token in attempts:
        ok, message = check(url, token)
        if ok:
            print(message)
            return 0
        errors.append(message)

    print("; ".join(errors), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
