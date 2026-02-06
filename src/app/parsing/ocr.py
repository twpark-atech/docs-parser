# ==============================================================================
# 목적 : OCR 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import base64
import logging
from typing import Optional

import requests

_log = logging.getLogger(__name__)

def _openai_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _auth_headers(api_key: Optional[str]) -> dict:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def openai_chat_image(
    *,
    base_url: str,
    model: str,
    api_key: Optional[str],
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> str:
    """OpenAI 호환 /v1/chat/completions로 이미지+프롬프트를 보내고 응답 텍스트를 반환합니다."""
    endpoint = _openai_endpoint(base_url)
    b64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(endpoint, json=payload, headers=_auth_headers(api_key), timeout=timeout_sec)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        _log.error("openai compat error: status=%s body=%s", r.status_code, r.text[:1000])
        raise
    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"openai compat returned empty choices: keys={list(data.keys())}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"openai compat returned empty content: keys={list(data.keys())}")
    return content


def ollama_chat_image(
    *,
    base_url: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> str:
    """Ollama /api/chat로 이미지+프롬프트를 보내고 응답 텍스트를 반환합니다.

    image_bytes를 base64로 인코딩해 Ollama Chat API에 포함시킵니다.
    stream=False로 단발 응답을 받아 message.content를 반환합니다.
    options.num_predict를 max_tokens로, option.temperature를 temperature로 전달합니다.

    Args:
        base_url: Ollama 서버 기본 URL.
        model: Ollama에 로드된 모델명.
        image_bytes: 입력 이미지 바이트.
        prompt: 사용자 프롬프트 문자열.
        max_tokens: 생성 최대 토큰 수.
        temperature: 샘플링 온도.
        timeout_sec: HTTP 요청 타임아웃(초).

    Returns:
        Ollama 응답의 message.content 문자열.

    Raises:
        requests.exceptions.RequestException: 네트워크/요청 레벨에서 오류가 발생할 경우.
        requests.HTTPError: 4xx/5xx 응답할 경우.
        ValueError: 응답 JSON 파싱에 실패할 경우.
        RuntimeError: 응답에 message.content가 없거나 빈 문자열인 경우.
    """
    endpoint = base_url.rstrip("/") + "/api/chat"
    b64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
    }

    r = requests.post(endpoint, json=payload, timeout=timeout_sec)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        _log.error("ollama /api/chat error: status=%s body=%s", r.status_code, r.text[:1000])
        raise
    data = r.json()

    msg = (data.get("message") or {})
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"ollama returned empty content: keys={list(data.keys())}")
    return content


def ollama_generate_image(
    *,
    base_url: str,
    model: str,
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> str:
    """Ollama /api/generate로 이미지+프롬프트를 보내고 응답 텍스트를 반환합니다."""
    endpoint = base_url.rstrip("/") + "/api/generate"
    b64 = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model,
        "stream": False,
        "prompt": prompt,
        "images": [b64],
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        }
    }

    r = requests.post(endpoint, json=payload, timeout=timeout_sec)
    try:
        r.raise_for_status()
    except requests.HTTPError:
        _log.error("ollama /api/generate error: status=%s body=%s", r.status_code, r.text[:1000])
        raise
    data = r.json()

    content = data.get("response")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"ollama returned empty response: keys={list(data.keys())}")
    return content


def ocr_page(
    image_bytes: bytes, 
    url: str, 
    model: str, 
    api_key: str, 
    prompt: str, 
    max_tokens: int, 
    temperature: float, 
    timeout_sec: int = 3600
) -> str:
    r"""페이지 이미지(PNG bytes)를 Ollama Chat API로 OCR/설명 처리하고 텍스트를 반환합니다.
    
    내부적으로 ollama_chat_image를 호출합니다.
    
    Args:
        image_bytes: 입력 이미지 바이트. 
        url: Ollama 서버 기본 URL.
        model: 사용할 Ollama 모델명.
        api_key: Authorization 토큰.
        prompt: OCR/설명용 프롬프트.
        max_tokens: 생성 최대 토큰 수.
        temperature: 샘플링 온도.
        timeout_sec: HTTP 요청 타임아웃(초).

    Returns:
        OCR/설명 결과 문자열.

    Raises:
        requests.exceptions.RequestException: 네트워크/요청 레벨에서 오류가 발생할 경우. 
        requests.HTTPError: 4xx/5xx 응답할 경우.
        ValueError: url 또는 model이 비어있을 경우.
        RuntimeError: Ollama 응답 content가 비어있을 경우.

    Examples:
        >>> text = ocr_page(png_bytes, "http://localhost:11434/v1/chat/completions", "gpt-oss:20b", "", "OCR 엔진으로서 이미지의 모든 텍스트를 추출해줘", 2048, 0.1, 3600)
        sub_title[[435, 111, 559, 135]]
        소재 개론
    """
    if not url or not model:
        raise ValueError("vlm.url and vlm.model are required.")
    
    try:
        if "/v1" in url:
            return openai_chat_image(
                base_url=url,
                model=model,
                api_key=api_key,
                image_bytes=image_bytes,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_sec=timeout_sec,
            )
        return ollama_chat_image(
            base_url=url,
            model=model,
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 400 and "/v1" not in url:
            return ollama_generate_image(
                base_url=url,
                model=model,
                image_bytes=image_bytes,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_sec=timeout_sec,
            )
        raise
