# ==============================================================================
# 목적 : OpenSearch 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Callable, Iterator

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk, scan as os_scan

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenSearchConfig:
    """OpenSearch 접속/인덱스 설정 클래스.
    OpenSearch Python 클라이언트 초기화에 필요한 URL과 인덱스명, 선택적 Basic Auth 및 TLS 인증서 검증 옵션을 담습니다.

    Attributes:
        url: OpenSearch 엔드포인트 URL.
        index: 기본 사용할 인덱스명.
        username: Basic Auth 사용자명.
        password: Basic Auth 비밀번호.
        verify_certs: TLS 인증서 검증 여부.
    """
    url: str
    index: str
    username: Optional[str] = None
    password: Optional[str] = None
    verify_certs: bool = False

IndexBody = Union[Dict[str, Any], Callable[[], Dict[str, Any]]]


class OpenSearchWriter:
    """OpenSearch 인덱스 생성/업서트/스캔을 수행하는 Writer 클래스.
    __init__: cfg.url을 host/port로 파싱해 OpenSearch 클라이언트를 생성합니다.
    ensure_index(): 인덱스가 없으면 지정 body로 생성합니다.
    bulk_upsert(): bulk API를 사용해 문서들을 배치로 index 합니다.
    scan(): helpers.scan(os_scan)로 query에 맞는 문서를 스트리밍 조회합니다.
    """
    def __init__(self, cfg: OpenSearchConfig):
        """Writer를 초기화하고 OpenSearch 클라이언트를 생성합니다.
        
        Args:
            cfg: OpenSearchConfig 설정 객체.

        Raises:
            ValueError: url에 포함된 port가 int 변환이 불가할 경우.
        """
        self._cfg = cfg
        host = cfg.url.replace("http://", "").replace("https://", "")
        if ":" in host:
            h, p = host.split(":", 1)
            port = int(p)
        else:
            h, port = host, 9200

        http_auth = None
        if cfg.username and cfg.password:
            http_auth = (cfg.username, cfg.password)

        self._client = OpenSearch(
            hosts=[{"host": h, "port": port}],
            http_auth=http_auth,
            use_ssl=cfg.url.startswith("https://"),
            verify_certs=cfg.verify_certs,
        )

    @property
    def index(self) -> str:
        return str(self._cfg.index)


    def ensure_index(self, *, body: Dict[str, Any]) -> None:
        """인덱스가 존재하도록 보장합니다.
        
        인덱스가 이미 존재하면 아무 작업도 하지 않습니다.
        존재하지 않으면 indices.create로 body를 사용해 인덱스를 생성합니다.

        Args:
            body: 인덱스 생성 요청 body.
        """
        if self._client.indices.exists(index=self._cfg.index):
            return
        self._client.indices.create(index=self._cfg.index, body=body)


    def bulk_upsert(self, docs: List[Dict[str, Any]], batch_size: int = 500) -> None:
        """문서 리스트를 bulk로 index합니다.
        
        docs의 각 원소는 {"_id": ..., "_source": ...} 형태를 기대하며, bulk 액션은 op_type="index"로 구성됩니다.
        동일 _id가 이미 존재하면 overwrite되고, 없으면 생성됩니다.

        batch_size 단위로 나누어 bulk(...)를 호출합니다.
        raise_on_error=False로 설정되어 일부 실패가 있더라도 전체 처리가 계속 진행되며, errors가 존재하면 일부만 warning 로그로 출력합니다.

        Args:
            docs: 업서트할 문서 리스트.
            batch_size: bulk 요청 배치 크기.

        Returns:
            None

        Raises:
            KeyError: docs 원소에 "_id" 또는 "_source"가 없을 경우.
        """
        actions = []
        for d in docs:
            actions.append(
                {
                    "_op_type": "index",
                    "_index": self._cfg.index,
                    "_id": d["_id"],
                    "_source": d["_source"],
                }
            )

        for i in range(0, len(actions), batch_size):
            chunk = actions[i : i + batch_size]
            success, errors = bulk(self._client, chunk, raise_on_error=False)
            if errors:
                _log.warning("OpenSearch bulk partial errors: %s", errors[:3])
            _log.info("OpenSearch bulk indexed: %d", success)

    
    def scan(self, *, query: Dict[str, Any], size: int = 500) -> Iterator[Dict[str, Any]]:
        """qeury에 매칭되는 문서를 scan(scroll)으로 순회합니다.
        
        helpers.scan(os_scan)을 사용해 결과를 스트리밍으로 yield 합니다.
        대량 조회 시 한 번에 모두 메모리에 올리지 않고 순회할 수 있습니다.

        Args:
            query: OpenSearch Query DSL dict.
            size: scroll batch size.

        Yield:
            OpenSerch hit dict.
        """
        for hit in os_scan(self._client, index=self._cfg.index, query=query, size=size):
            yield hit
