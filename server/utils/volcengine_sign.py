"""
Volcengine API signature v4 (HMAC-SHA256) implementation.
Compatible with the volcengine universal SDK pattern used in Go.
"""
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

_ALGORITHM = "HMAC-SHA256"
_ENDPOINT = "https://open.volcengineapi.com"


def _hmac_sha256(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def _hex_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _signing_key(sk: str, date: str, region: str, service: str) -> bytes:
    k = _hmac_sha256(("SDK_REQUEST" + sk).encode("utf-8"), date)
    k = _hmac_sha256(k, region)
    k = _hmac_sha256(k, service)
    k = _hmac_sha256(k, "request")
    return k


def _build_auth_header(
    ak: str,
    sk: str,
    region: str,
    service: str,
    method: str,
    path: str,
    query: str,
    headers: Dict[str, str],
    body_bytes: bytes,
    now: datetime,
) -> str:
    date_stamp = now.strftime("%Y%m%d")
    datetime_stamp = now.strftime("%Y%m%dT%H%M%SZ")

    # Canonical headers (sorted)
    signed_header_keys = sorted(k.lower() for k in headers)
    canonical_headers = "".join(
        f"{k}:{headers[next(hk for hk in headers if hk.lower() == k)]}\n"
        for k in signed_header_keys
    )
    signed_headers_str = ";".join(signed_header_keys)

    payload_hash = _hex_sha256(body_bytes)

    canonical_request = "\n".join([
        method.upper(),
        path or "/",
        query,
        canonical_headers,
        signed_headers_str,
        payload_hash,
    ])

    credential_scope = f"{date_stamp}/{region}/{service}/request"
    string_to_sign = "\n".join([
        _ALGORITHM,
        datetime_stamp,
        credential_scope,
        _hex_sha256(canonical_request.encode("utf-8")),
    ])

    sig = hmac.new(
        _signing_key(sk, date_stamp, region, service),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return (
        f"{_ALGORITHM} Credential={ak}/{credential_scope},"
        f"SignedHeaders={signed_headers_str},Signature={sig}"
    )


async def volcengine_post(
    ak: str,
    sk: str,
    service: str,
    action: str,
    version: str,
    body: Dict[str, Any],
    region: str = "cn-beijing",
) -> Dict[str, Any]:
    """Call a volcengine OpenAPI action via POST."""
    now = datetime.now(timezone.utc)
    datetime_stamp = now.strftime("%Y%m%dT%H%M%SZ")

    body_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    query = f"Action={action}&Version={version}"
    url = f"{_ENDPOINT}/?{query}"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Host": "open.volcengineapi.com",
        "X-Date": datetime_stamp,
        "X-Content-Sha256": _hex_sha256(body_bytes),
    }

    headers["Authorization"] = _build_auth_header(
        ak=ak,
        sk=sk,
        region=region,
        service=service,
        method="POST",
        path="/",
        query=query,
        headers=headers,
        body_bytes=body_bytes,
        now=now,
    )

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, content=body_bytes, headers=headers)
        resp.raise_for_status()
        return resp.json()
