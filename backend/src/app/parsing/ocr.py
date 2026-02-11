# ==============================================================================
# 목적 : OCR 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import base64
import logging

from openai import OpenAI

_log = logging.getLogger(__name__)


def litellm_chat_image(
    *,
    lm_client: OpenAI,
    model: str,
    image_bytes: bytes,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> str:
    """Lite LLM으로 이미지+프롬프트를 보내고 응답 텍스트를 반환합니다.

    image_bytes를 base64로 인코딩해 Lite LLM API에 포함시킵니다.
    stream=False로 단발 응답을 받아 message.content를 반환합니다.
    options.num_predict를 max_tokens로, option.temperature를 temperature로 전달합니다.

    Args:
        base_url: Lite LLM 서버 기본 URL.
        model: Lite LLM 로드된 모델명.
        image_bytes: 입력 이미지 바이트.
        prompt: 사용자 프롬프트 문자열.
        max_tokens: 생성 최대 토큰 수.
        temperature: 샘플링 온도.
        timeout_sec: HTTP 요청 타임아웃(초).

    Returns:
        Lite LLM 응답의 message.content 문자열.

    Raises:
        requests.exceptions.RequestException: 네트워크/요청 레벨에서 오류가 발생할 경우.
        requests.HTTPError: 4xx/5xx 응답할 경우.
        ValueError: 응답 JSON 파싱에 실패할 경우.
        RuntimeError: 응답에 message.content가 없거나 빈 문자열인 경우.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        completion = lm_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_sec
        )
    except TimeoutError as e:
        raise f"OCR 요청이 {timeout_sec}초 내에 완료되지 않았ㅅ브니다."
    except Exception as e:
        raise f"OCR 엔진 처리 중 오류가 발생했습니다: {e!s}" from e
    
    content = completion.choices[0].message.content
    if not content:
        raise "OCR 엔진이 빈 응답을 반환했습니다."
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
    """페이지 이미지(PNG bytes)를 Ollama Chat API로 OCR/설명 처리하고 텍스트를 반환합니다.
    
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
    
    lm_client = OpenAI(base_url=url, api_key=api_key)

    try:
        return litellm_chat_image(
            lm_client=lm_client,
            model=model,
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec
        )
    except Exception as e:
        raise f"Lite LLM Error: {e}"