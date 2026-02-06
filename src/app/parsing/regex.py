# ==============================================================================
# 목적 : 정규식 관련 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import re

RE_IMG_BOX = re.compile(r"image\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]")
RE_CAP_BOX = re.compile(r"image_caption\[\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\]")
RE_REF_DET = re.compile(
    r"<\|ref\|>\s*(?P<kind>[^<]+?)\s*<\|/ref\|>\s*"
    r"<\|det\|>\s*\[\[\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x2>-?\d+(?:\.\d+)?)\s*,\s*(?P<y2>-?\d+(?:\.\d+)?)"
    r"\s*\]\]\s*<\|/det\|>",
    re.MULTILINE,
)
RE_IMG_TOKEN = re.compile(r"\[\[IMG:([^\]]+)\]\]")
RE_PAGE_HEADER = re.compile(r"(?m)^\s*##\s*Page\s+(\d+)\s*$")
RE_HTML_TABLE = re.compile(
    r"("
    r"<table\b[^>]*>.*?</table>"
    r"|"
    r"^\s*table\[\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\]\s*"
    r"(?:\n.*?)*?"
    r")(?=^\s*(?:text|sub_title|image_caption|table_caption|table|image|list_item)\s*\[\[|^---|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
RE_SENT_END = re.compile(r"[.!?](?=\s|$)")

RE_BULLET = re.compile(r"^(\s*)([-•●▪◦])\s+(.*)$")
RE_NUM = re.compile(r"^(\s*)(\d+)[\.\)]\s+(.*)$")
RE_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
RE_FILENAME = re.compile(r"[^a-zA-Z0-9._-]+")
RE_IMAGE_REF_DET = re.compile(
    r"<\|ref\|>image<\|/ref\|>\s*<\|det\|>\s*"
    r"\[?\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\]?\s*"
    r"<\|/det\|>",
    re.IGNORECASE | re.MULTILINE,
)
RE_KIND_BBOX = re.compile(
    r"\b(?P<kind>text|sub_title|image_caption|table_caption|table|image|list_item)\s*"
    r"\[\[\s*(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\]\]\s*",
    re.IGNORECASE,
)
RE_TABLE_CAPTION = re.compile(r"^\s*표\s*\d+\s*[\.\)]\s*.*$")
RE_SENT = re.compile(r".+?(?:[\.!?]+|\n+|$)", re.DOTALL)
RE_REFDET_LINE = re.compile(r"^(?P<prefix>.*?)<\|det\|>(?P<payload>.*?)<\|/det\|>(?P<tail>.*)$", re.IGNORECASE)
RE_TAG = re.compile(r"<\|/?[a-zA-Z0-9_]+\|>")
RE_MULTI_WS = re.compile(r"[ \t]+")
RE_HTML_TAG = re.compile(r"</?[^>]+>", re.IGNORECASE)
RE_PIPE_TAG = re.compile(r"<\|/?[a-zA-Z0-9_:\-]+\|>")
RE_DET_BLOCK = re.compile(r"<\|det\|>(.*?)<\|/det\|>", re.IGNORECASE | re.DOTALL)
RE_TEXT_BBOX = re.compile(
    r"\b(?:text|sub_title|image_caption|table|image|list_item)\s*"
    r"\[\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]\]\s*",
    re.IGNORECASE,
)
RE_ANY_BBOX_CAPTURE = re.compile(
    r"\[\[\s*(?P<x0>-?\d+(?:\.\d+)?)\s*,\s*(?P<y0>-?\d+(?:\.\d+)?)\s*,\s*(?P<x1>-?\d+(?:\.\d+)?)\s*,\s*(?P<y1>-?\d+(?:\.\d+)?)\s*\]\]"
)
RE_REFDET_BLOCK = re.compile(
    r"<\|ref\|>(?P<ref>.*?)<\|/ref\|>\s*<\|det\|>(?P<det>.*?)<\|/det\|>",
    re.IGNORECASE | re.DOTALL,
)

HNSW_4096 = {
    "type": "knn_vector",
    "dimension": 4096,
    "method": {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil"
    },
}

HNSW_1024 = {
    "type": "knn_vector",
    "dimension": 1024,
    "method": {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil"
    },
}
