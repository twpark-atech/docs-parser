# ==============================================================================
# 목적 : 이미지 추출 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import re, io, json, hashlib, logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

from app.common.hash import sha256_bytes
from app.parsing.ocr import ocr_page
from app.parsing.regex import (
    RE_CAP_BOX,
    RE_REF_DET,
    RE_SENT_END,
    RE_IMAGE_REF_DET,
    RE_ANY_BBOX_CAPTURE,
    RE_KIND_BBOX,
)

_log = logging.getLogger(__name__)


def _make_image_id(doc_id: str, page_no: int, bbox: Tuple[int, int, int, int]) -> str:
    """문서/페이지/바운딩박스 기반으로 안정적인 이미지 ID를 생성합니다.
    
    동일한 (doc_id, page_no, bbox) 조합에 대해 항상 동일한 image_id를 반환합니다.
    bbox는 최종 이미지 bbox(정수 좌표, 패딩 미포함/포함 여부는 호출부 정책)에 해당합니다.
    
    Args:
        doc_id: 문서 고유 ID.
        page_no: 0-based 또는 파이프라인에서 사용하는 페이지 번호.
        bbox: (x1, y1, x2, y2) 정수 좌표.
        
    Returns:
        생성된 image_id 문자열. 형식: "{doc_id}:p{page_no}:img{hash16}".
    """
    key = f"{doc_id}|p{page_no}|{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}:p{page_no}:img{h}"


def _infer_bbox_scale(boxes: List[Tuple[float, float, float, float]], W: int, H: int) -> Tuple[float, float]:
    """OCR/Det 결과 bbox 좌표계가 이미지 픽셀 좌표인지 추론해 스케일을 계산합니다.
    
    bbox 값의 범위를 기반으로 다음 케이스를 휴리스틱하게 처리합니다.
    - 정규화 좌표(0~1.x): (sx, sy) = (W, H)
    - 0~1000 근처 좌표인데 실제 이미지가 더 큰 경우: 1000 기준으로 스케일 업
    - 그 외: max 좌표와 이미지 크기의 비율로 스케일 추정

    Args:
        boxes: (x1, y1, x2, y2) 리스트.
        W: 원본 페이지 이미지 너비(px).
        H: 원본 페이지 이미지 높이(px).

    Returns:
        (sx, sy) 스케일 팩터.
        원본 bbox 좌표에 곱하면 이미지 픽셀 좌표로 매핑됩니다.
    """
    if not boxes:
        return 1.0, 1.0
    
    max_x2 = max(b[2] for b in boxes)
    max_y2 = max(b[3] for b in boxes)
    
    if max_x2 <= 1.5 and max_y2 <= 1.5:
        return float(W), float(H)
    
    if max_x2 <= 1100 and max_y2 <= 1100 and (W > 1200 or H > 1200):
        return H / 1000.0, H / 1000.0
    
    sx = 1.0
    sy = 1.0
    if max_x2 > 0 and (max_x2 < W * 0.85 or max_x2 > W * 1.15):
        sx = W / float(max_x2)
    if max_y2 > 0 and (max_y2 < H * 0.85 or max_y2 > H * 1.15):
        sy = H / float(max_y2)

    return sx, sy


def _transform_bbox(
    x1: float, y1: float, x2: float, y2: float,
    *, sx: float, sy: float, W: int, H: int,
    pad_ratio: float,
) -> Tuple[int, int, int, int]:
    """bbox를 이미지 픽셀 좌표로 변환하고 패딩/클램프를 적용합니다.

    - 입력 bbox에 (sx, sy)를 곱해 픽셀 좌표로 변환합니다.
    - bbox 크기에 비례한 pad_ratio만큼 상하좌우를 확장합니다.
    - 결과는 [0..W], [0..H] 범위로 클램프 됩니다.

    Args:
        x1,y1,x2,y2: 원본 bbox 좌표(정규화 또는 임의 스케일).
        sx, sy: 좌표 변환 스케일.
        W, H: 이미지 크기(px).
        pad_ratio: bbox 너비/높이에 대한 패딩 비율(0.0이면 패딩 없음).

    Returns:
        (fx1, fy1, fx2, fy2) 정수 픽셀 좌표 bbox.    
    """
    fx1 = int(round(x1 * sx))
    fy1 = int(round(y1 * sy))
    fx2 = int(round(x2 * sx))
    fy2 = int(round(y2 * sy))

    bw = max(1, fx2 - fx1)
    bh = max(1, fy2 - fy1)
    px = int(round(bw * pad_ratio))
    py = int(round(bh * pad_ratio))

    fx1 -= px
    fy1 -= py
    fx2 += px
    fy2 += py

    fx1 = max(0, min(fx1, W))
    fx2 = max(0, min(fx2, W))
    fy1 = max(0, min(fy1, H))
    fy2 = max(0, min(fy2, H))
    return fx1, fy1, fx2, fy2


def _cut_by_sentence_end(s: str, max_chars: int) -> str:
    """문자열을 최대 길이로 자르되 문장 끝(정규식) 기준을 우선합니다.
    
    max_chars 이전 구간에서 문장 종료 패턴(RE_SENT_END)을 찾아 가장 마지막 종료 지점까지 잘라 반환합니다.
    종료 지점이 없으면 단순 절단합니다.

    Args:
        s: 원본 문자열.
        max_chars: 최대 허용 문자 수.

    Returns:
        잘린 문자열(우측 공백 제거).
    """
    if len(s) <= max_chars:
        return s
    
    prefix = s[:max_chars]
    last_end = -1
    for m in RE_SENT_END.finditer(prefix):
        last_end = m.end()

    if last_end > 0:
        return prefix[:last_end].rstrip()
    
    return prefix.rstrip()


def describe_image_short(
    image_bytes: bytes,
    *,
    vlm_url: str,
    vlm_model: str,
    vlm_api_key: str,
    timeout_sec: int,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """이미지에 대한 짧은 설명 텍스트를 생성합니다(주로 OCR/VLM 기반).

    내부적으로 ocr_page를 호출해 텍스트를 얻고, 공백 정규화 및 문장 단위 컷(_cut_by_sentence_end)를 적용합니다.
    실패 시 예외를 로깅하고 빈 문자열을 반환합니다.

    Args:
        image_bytes: PNG/JPEG 등 이미지 바이너리.
        vlm_url: VLM/OCR 서버 URL.
        vlm_model: 사용할 모델 ID.
        vlm_api_key: API 키(필요 시).
        timeout_sec: 요청 타임아웃(초).
        prompt: 모델 프롬프트.
        max_tokens: 생성 최대 토큰 수.
        temperature: 생성 다양성 파라미터.

    Returns:
        정제된 설명 문자열. 실패/결과 없음이면 "".
    """
    try:
        txt = ocr_page(
            image_bytes,
            vlm_url, vlm_model, vlm_api_key,
            prompt,
            max_tokens, temperature, timeout_sec,
        )

        if not txt:
            return ""
        
        s = " ".join(txt.split()).strip()
        if not s:
            return ""
        
        s = _cut_by_sentence_end(s, max_chars=1200)
        return s
    
    except Exception as e:
        _log.exception("IMG_DESC failed: %s", e)

    return ""


def load_desc_cache(desc_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """이미지 설명 캐시(JSONL)를 로드합니다.
    
    JSONL 각 라인의 주요 키를 사용합니다.
    - image_sha256: 이미지 바이트 SHA-256
    - description: 이미지 설명 텍스트
    - image_id: 이미지 ID

    Args:
        desc_jsonl: 캐시 파일 경로(jsonl).

    Returns:
        {image_sha256: {"description": str, "image_id": str}} 형태의 캐시 딕셔너리.
        파일이 없거나 파싱 실패 시 빈 dict를 반환합니다.    
    """
    cache: Dict[str, Dict[str, str]] = {}
    if not desc_jsonl.exists():
        return cache
    
    try:
        with desc_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                img_sha = r.get("image_sha256")
                desc = r.get("description")
                if img_sha and desc:
                    cache[img_sha] = {"description": str(desc), "image_id": r.get("image_id", "")}
    except Exception as e:
        _log.warning("Failed to load image desc cache: %s", e)
    
    return cache


def extract_and_store_images_from_page(
    page_png_path: Path,
    ocr_text: str,
    *,
    assets_root: Path,
    doc_id: str,
    doc_title: str,
    source_uri: str,
    sha256: str,
    page_no: int,
    pad_ratio: float,
    debug_draw_bboxes: bool,
    do_image_desc: bool,
    img_desc_prompt: str,
    img_desc_max_tokens: int,
    img_desc_temperature: float,
    vlm_url: str,
    vlm_model: str,
    vlm_api_key: str,
    vlm_timeout_sec: int,
    desc_cache: Dict[str, Dict[str, str]],
    filter_enabled: bool = False,
    stddev_min: float = 0.0,
    stddev_mode: str = "grayscale",
    skip_fullpage: bool = True,
    rewrite_det: bool = False,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    """페이지 OCR 텍스트의 det bbox를 기반으로 이미지를 추출/저장하고 메타/설명을 생성합니다.

    처리 흐름:
    1) ocr_text에서 "image det bbox" 패턴(RE_REF_DET)을 매칭합니다.
    2) bbox 좌표계를 추론(_infer_bbox_scale)하여 이미지 픽셀 좌표로 변환합니다.
    3) crop_bbox(패딩 포함)로 이미지를 잘라 PNG로 저장하고, 메타(json)을 기록합니다.
    4) 옵션(do_image_desc=True)인 경우 이미지 설명을 생성/캐시합니다.
    5) 옵션(rewrite_det=True)인 경우 텍스트 내 det bbox를 image_id로 치환합니다.

    Args:
        page_png_path: 페이지 전체 이미지(PNG) 경로.
        ocr_text: bbox 태그가 포함된 OCR 결과 텍스트(원본이어야 함).
        assets_root: 산출물 루트 디렉토리.
        doc_id: 문서 고유 ID.
        doc_title: 문서 제목.
        source_uri: 원본 소스 URI.
        sha256: 문서 sha256.
        page_no: 페이지 번호.
        pad_ratio: crop bbox 확장 비율.
        debug_draw_bboxes: True이면 bbox 시각화 이미지를 저장합니다.
        do_image_desc: True이면 이미지 설명(텍스트)을 생성합니다.
        img_desc_prompt: 설명 생성에 사용할 프롬프트.
        img_desc_max_tokens: 설명 생성 최대 토큰.
        img_desc_temperature: 설명 생성 다양성 파라미터.
        vlm_url: VLM 호출 URL.
        vlm_model: VLM 호출 모델명.
        vlm_api_key: VLM 호출 API Key.
        vlm_timeout_sec: VLM 호출 타임아웃(초).
        desc_cache: {image_sha256: {"description": str, "image_id": str}} 캐시.
        filter_enabled: True이면 stddev 기반 필터링을 적용합니다.
        stddev_min: 필터링 기준 최소 표준편차 값.
        stddev_mode: "grayscale" 또는 "rgb" 표준편차 계산 모드.
        skip_fullpage: True면 전체 페이지 bbox는 스킵합니다.
        rewrite_det: True면 ocr_text 내 det bbox를 image_id로 치환환 텍스트를 반환합니다.

    Returns:
        Tuple of:
            - new_text: rewrite_det=True이면 치환된 텍스트, 아니면 원본 oct_text.
            - img_records: 추출된 이미지 메타 레코드 리스트.
            - desc_records: 생성된 이미지 설명 레코드 리스트.
            - det_bbox_to_image_id: "dl,dt,dr,db" -> image-Id 매핑.
    """
    def _g(m: re.Match, *names: str, default: str = "") -> str:
        gd = m.groupdict() or {}
        for n in names:
            v = gd.get(n)
            if v is not None:
                return str(v)
        return default

    if not ocr_text:
        return ocr_text, [], [], {}
    
    img_dir = assets_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    page_img = Image.open(page_png_path).convert("RGB")
    W, H = page_img.size

    lines = ocr_text.splitlines()
    cap_candidates: List[Tuple[Tuple[float, float, float, float], str]] = []
    for i, ln in enumerate(lines):
        mcap = RE_CAP_BOX.search(ln)
        if not mcap:
            continue
        try:
            cx1, cy1, cx2, cy2 = map(float, mcap.groups())
        except Exception:
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        cap_text = lines[j].strip() if j < len(lines) else ""
        cap_candidates.append(((cx1, cy1, cx2, cy2), cap_text))

    det_matches = list(RE_REF_DET.finditer(ocr_text))
    img_matches = [m for m in det_matches if _g(m, "kind", "ref", "tag").strip() == "image"]

    if not img_matches:
        kind_matches = list(RE_KIND_BBOX.finditer(ocr_text))
        img_matches = [m for m in kind_matches if (m.group("kind") or "").lower() == "image"]

    if not img_matches:
        return ocr_text, [], [], {}
    
    raw_boxes: List[Tuple[float, float, float, float]] = []
    for m in img_matches:
        try:
            if "x2" in m.groupdict() and "y2" in m.groupdict():
                raw_boxes.append(
                    (
                        float(m.group("x1")),
                        float(m.group("y1")),
                        float(m.group("x2")),
                        float(m.group("y2")),
                    )
                )
            else:
                raw_boxes.append(
                    (
                        float(m.group("x0")),
                        float(m.group("y0")),
                        float(m.group("x1")),
                        float(m.group("y1")),
                    )
                )
        except Exception:
            continue

    if not raw_boxes:
        _log.info("Image bbox matched but parse failed. page_no=%s", page_no)

    sx, sy = _infer_bbox_scale(raw_boxes, W, H)

    debug_img = page_img.copy()
    debug_draw = ImageDraw.Draw(debug_img) if debug_draw_bboxes else None

    img_records: List[Dict[str, Any]] = []
    desc_records: List[Dict[str, Any]] = []
    new_text = ocr_text

    det_bbox_to_image_id: Dict[str, str] = {}

    for m in img_matches:
        try:
            if "x2" in m.groupdict() and "y2" in m.groupdict():
                x1 = float(m.group("x1"))
                y1 = float(m.group("y1"))
                x2 = float(m.group("x2"))
                y2 = float(m.group("y2"))
            else:
                x1 = float(m.group("x0"))
                y1 = float(m.group("y0"))
                x2 = float(m.group("x1"))
                y2 = float(m.group("y1"))
        except Exception:
            continue

        if skip_fullpage:
            if abs(x1) < 1e-6 and abs(y1) < 1e-6 and x2 >= 0.99 and y2 >= 0.99:
                continue
            if abs(x1) < 1e-6 and abs(y1) < 1e-6 and x2 >= 999 and y2 >= 999:
                continue

        dl = int(round(x1))
        dt = int(round(y1))
        dr = int(round(x2))
        db = int(round(y2))
        det_key = f"{dl},{dt},{dr},{db}"

        rx1, ry1, rx2, ry2 = _transform_bbox(x1, y1, x2, y2, sx=sx, sy=sy, W=W, H=H, pad_ratio=0.0)
        cx1, cy1, cx2, cy2 = _transform_bbox(x1, y1, x2, y2, sx=sx, sy=sy, W=W, H=H, pad_ratio=pad_ratio)

        if rx2 <= rx1 or ry2 <= ry1:
            continue
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        bbox_for_id = (rx1, ry1, rx2, ry2)
        image_id =_make_image_id(doc_id, page_no, bbox_for_id)

        det_bbox_to_image_id.setdefault(det_key, image_id)

        safe_name = image_id.replace(":", "_")
        img_path = img_dir / f"{safe_name}.png"
        meta_path = img_dir / f"{safe_name}.json"

        if not img_path.exists():
            crop = page_img.crop((cx1, cy1, cx2, cy2))

            should_skip, stddev = should_skip_crop_by_stddev(
                crop,
                enabled=filter_enabled,
                stddev_min=stddev_min,
                mode=stddev_mode,
            )
            if should_skip:
                continue

            crop.save(img_path, format="PNG")

        with Image.open(img_path) as _ci:
            cw, ch = _ci.size

        cap_text_best = ""
        if cap_candidates:
            ix, iy = (cx1 + cx2) / 2, (cy1 + cy2) / 2
            best_d = None
            for (bx1, by1, bx2, by2), cap_text in cap_candidates:
                bx1i, by1i, bx2i, by2i = _transform_bbox(bx1, by1, bx2, by2, sx=sx, sy=sy, W=W, H=H, pad_ratio=0.0)
                cx, cy = (bx1i + bx2i) / 2, (by1i + by2i) / 2
                d = (ix - cx) ** 2 + (iy - cy) ** 2
                if best_d is None or d < best_d:
                    best_d = d
                    cap_text_best = cap_text

        record = {
            "image_id": image_id,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "source_uri": source_uri,
            "sha256": sha256,
            "page_no": page_no,
            "det_bbox": {"x1": dl, "y1": dt, "x2": dr, "y2": db},
            "bbox": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
            "crop_bbox": {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2},
            "bbox_transform": {"sx": sx, "sy": sy, "pad_ratio": pad_ratio},
            "image_path": str(img_path),
            "width": int(cw),
            "height": int(ch),
            "caption": cap_text_best,
        }
        meta_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        img_records.append(record)

        if do_image_desc:
            b = img_path.read_bytes()
            image_sha = sha256_bytes(b)
            desc_obj = desc_cache.get(image_sha)
            desc_text = (desc_obj or {}).get("description", "") if isinstance(desc_obj, dict) else ""
            if not desc_text:
                desc_text = describe_image_short(
                    b,
                    vlm_url=vlm_url,
                    vlm_model=vlm_model,
                    vlm_api_key=vlm_api_key,
                    timeout_sec=vlm_timeout_sec,
                    prompt=img_desc_prompt,
                    max_tokens=img_desc_max_tokens,
                    temperature=img_desc_temperature,
                )
                if desc_text:
                    desc_cache[image_sha] = {"description": desc_text, "image_id": image_id}

            if desc_text:
                desc_records.append({
                    "image_id": image_id,
                    "image_sha256": image_sha,
                    "doc_id": doc_id,
                    "page_no": page_no,
                    "image_path": str(img_path),
                    "image_mime": "image/png",
                    "width": int(cw),
                    "height": int(ch),
                    "caption": cap_text_best,
                    "description": desc_text,
                })
        
        if debug_draw is not None:
            debug_draw.rectangle([cx1, cy1, cx2, cy2], width=3)

    if debug_draw_bboxes:
        debug_path = assets_root / "pages" / f"p{page_no:04d}.bboxes.png"
        debug_img.save(debug_path, format="PNG")

    if rewrite_det and det_bbox_to_image_id:
        new_text = rewrite_image_det_bbox_to_image_id(new_text, det_bbox_to_image_id)

    return new_text, img_records, desc_records, det_bbox_to_image_id


def should_skip_crop_by_stddev(
    crop: Image.Image,
    *,
    enabled: bool,
    stddev_min: float,
    mode: str = "grayscale",
) -> Tuple[bool, float]:
    """crop 이미지의 표준편차(stddev)로 '의미 없는 이미지'를 스킵할지 판단합니다.

    배경만 있거나 단색에 가까운 영역은 표준편차가 작게 나오므로 stddev_min 미만이면 스킵 대상으로 판단합니다.

    Args:
        crop: 잘라낸 이미지(PIL Image).
        enabled: 필터링 사용 여부.
        stddev_min: 스킵 판정 기준(이 값 미만이면 스킵).
        mode: "grayscale" 또는 "rgb".

    Returns:
        (should_skip, stddev)
        - should_skip: True면 스킵 권장.
        - stddev: 계산된 표준편차 값.
    """
    if not enabled or stddev_min <= 0:
        return False, 0.0
    
    if mode not in ("grayscale", "rgb"):
        raise ValueError(f"mode must be 'grayscale' or 'rgb'. got={mode}")
    
    if mode == "grayscale":
        arr = np.asarray(crop.convert("L"), dtype=np.float32)
        stddev = float(arr.std())
        return stddev < stddev_min, stddev
    
    arr = np.asarray(crop.convert("RGB"), dtype=np.float32)
    stddev = float(arr.std())
    return stddev < stddev_min, stddev


def get_image_size_from_bytes(image_bytes: bytes) -> Tuple[int, int]:
    """이미지 바이트로부터 (width, height)를 추출합니다.

    Args:
        image_bytes: 이미지 바이너리(PNG/JPEG 등).

    Returns:
        (width, height). 파싱 실패 시 (0, 0).
    """
    try:
        im = Image.open(io.BytesIO(image_bytes))
        return int(im.size([0])), int(im.size[1])
    except Exception:
        return 0, 0
    

def rewrite_image_det_bbox_to_image_id(md_or_text: str, bbox_to_image_id: Dict[str, str]) -> str:
    """텍스트 내 image det bbox를 iamge_id 참조 형태로 치환합니다.

    RE_IMAGE_REF_DET로 매칭되는 bbox(l,t,r,b)를 정수로 정규화합니다.
    bbox_to_image_id에서 image_id를 찾아 다음 형태로 치환합니다.
    - "<|ref|>image<|/ref|><|det|>{image_id}<|/det|>"

    Args:
        md_or_text: 원본 텍스트 또는 마크다운.
        bbox_to_image_id: "l,t,r,b" -> image_id 매핑.

    Returns:
        치환된 텍스트. 매핑이 없으면 원본 매치 문자열을 그대로 유지합니다.
    """
    def repl(m: re.Match) -> str:
        l = int(float(m.group(1)))
        t = int(float(m.group(2)))
        r = int(float(m.group(3)))
        b = int(float(m.group(4)))
        key = f"{l},{t},{r},{b}"
        image_id = bbox_to_image_id.get(key)
        if not image_id:
            return m.group(0)
        return f"<|ref|>image<|/ref|><|det|>{image_id}<|/det|>"
    
    out = RE_IMAGE_REF_DET.sub(repl, md_or_text)

    def repl_kind(m: re.Match) -> str:
        kind = (m.group("kind") or "").lower()
        if kind != "image":
            return m.group(0)
        l = int(float(m.group("x0")))
        t = int(float(m.group("y0")))
        r = int(float(m.group("x1")))
        b = int(float(m.group("y1")))
        key = f"{l},{t},{r},{b}"
        image_id = bbox_to_image_id.get(key)
        if not image_id:
            return m.group(0)
        return f"<|ref|>image<|/ref|><|det|>{image_id}<|/det|>"

    return RE_KIND_BBOX.sub(repl_kind, out)
