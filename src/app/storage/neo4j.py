# ==============================================================================
# 목적 : Neo4j 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Any

from neo4j import GraphDatabase

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j 접속 설정 클래스
    Neo4j Python Driver(GraphDatabase.driver) 초기화에 필요한 접속 URI와 인증 정보를 보관합니다.

    Attributes:
        uri: Neo4j 접속 URI.
        username: 사용자명.
        password: 비밀번호.
        database: 사용할 DB 이름.
    """
    uri: str
    username: str
    password: str
    database: str


class Neo4jWriter:
    """Neo4j에 Document/Chunk 그래프를 저장하는 Writer 클래스.
    
    ensure_constraints(): Document.doc_id, Chunk.chunk_id에 대한 유니크 제약을 생성.
    upsert_document_and_chunks(): Document 및 Chunk 노드를 MERGE로 업서트합니다.
        (Document)-[:HAS_CHUNK]->(Chunk) 관계 및 (Chunk)-[:NEXT]->(Chunk) 체인을 구성합니다.

    그래프 모델
    - (:Document {doc_id, title, source_uri, sha256, updated_at})
    - (:Chunk {chunk_id, doc_id, page_start, page_end, order})
    - (Document)-[:HAS_CHUNK]->(Chunk)
    - (Chunk)-[:NEXT]->(Chunk)
    """
    def __init__(self, cfg: Neo4jConfig):
        """Writer를 초기화합니다.

        Args:
            cfg: Neo4jconfig 설정 객체.
        """
        self._cfg = cfg
        self._driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))

    def close(self) -> None:
        """Neo4j 드라이버 리소스를 정리합니다.
        
        커넥션 풀/소켓 등을 닫기 위해 프로세스 종료 전 호출하는 것이 안전합니다.
        """
        self._driver.close()

    def ensure_constraints(self) -> None:
        """Document/Chunk 유니크 제약 조건을 생성합니다.
        
        생성하려는 제약
        - (d:Document) d.doc_id UNIQUE
        - (c:Chunk) c.chunk_id UNIQUE
        
        IF NOT EXISTS를 사용해 이미 존재하면 no-op가 되도록 했습니다.
        실행 중 예외가 발생해도 파이프라인을 막지 않기 위해 예외를 잡고 info 로그만 남깁니다.
        """
        stmts = [
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
        ]
        with self._driver.session(database=self._cfg.database) as session:
            for s in stmts:
                try:
                    session.run(s).consume()
                except Exception as e:
                    _log.info("Constraint creation skipped/failed: %s", e)

    def upsert_document_and_chunks(
        self,
        doc: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        *,
        rebuild_next: bool = True,
    ) -> None:
        """Document 및 Chunk를 업서트하고 HAS_CHUNK/NEXT 관계를 생성합니다.
        
        1) MERGE로 Document(doc_id) 노드를 업서트하고 문서 메타(title/source_uri/sha256/updated_at)를 SET합니다.
        2) UNWIND로 chunks를 순회하며 MERGE로 Chunk(chunk_id) 노드를 업서트하고 doc_id/page_start/page_end/order를 SET합니다.
        3) (Document)-[:HAS_CHUNK]->(Chunk) 관계를 MERGE로 보장합니다.
        4) chunks를 order 기준으로 정렬한 뒤 인접 쌍을 만들어 (Chunk)-[:NEXT]->(Chunk) 관계를 MERGE합니다.

        Args:
            doc: 문서 메타 dict.
            - 최소 키: doc_id, title, source_uri, sha256
            chunks: 청크 메타 dict 리스트.
            - 최소 키: chunk_id, doc_id, page_start, page_end, order
            rebuild_next: True면 해당 doc_id의 기존 NEXT 관계를 삭제 후 재생성합니다.

        Returns:
            None

        Raises:
            KeyError: chunks 정렬 시 "order" 또는 관계 생성 시 "chunk_id" 등이 누락된 경우.
            neo4j.exception.Neo4jError: Cypher 실행에 실패한 경우.
        """
        if not chunks:
            with self._driver.session(database=self._cfg.database) as session:
                session.execute_write(self._tx_upsert_document_only, doc)
            return
        
        with self._driver.session(database=self._cfg.database) as session:
            session.execute_write(self._tx_upsert_document_and_chunks, doc, chunks)

            if rebuild_next:
                session.execute_write(self._tx_delete_next_for_doc, doc["doc_id"])

            ordered = sorted(
                chunks,
                key=lambda x: (int(x.get("page_start", 0)), int(x.get("order", 0))),
            )
            pairs = [
                {"a": ordered[i]["chunk_id"], "b": ordered[i + 1]["chunk_id"]}
                for i in range(len(ordered) - 1)
                if ordered[i].get("chunk_id") and ordered[i + 1].get("chunk_id")
            ]
            if pairs:
                session.execute_write(self._tx_upsert_next_pairs, pairs)

    @staticmethod
    def _tx_upsert_document_only(tx, doc: Dict[str, Any]) -> None:
        cypher = """
        MERGE (d:Document {doc_id: $doc.doc_id})
        SET d.title = $doc.title,
            d.source_uri = $doc.source_uri,
            d.sha256 = $doc.sha256,
            d.updated_at = datetime()
        """
        tx.run(
            cypher,
            doc_id=doc.get("doc_id"),
            title=doc.get("title"),
            source_uri=doc.get("source_uri"),
            sha256=doc.get("sha256"),
        ).consume()

    @staticmethod
    def _tx_upsert_document_and_chunks(tx, doc: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        cypher = """
        MERGE (d:Document {doc_id: $doc.doc_id})
        SET d.title = $doc.title,
            d.source_uri = $doc.source_uri,
            d.sha256 = $doc.sha256,
            d.updated_at = datetime()
        WITH d
        UNWIND $chunks AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c.doc_id = row.doc_id,
            c.page_start = row.page_start,
            c.page_end = row.page_end,
            c.order = row.order
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN count(*) AS n
        """
        tx.run(cypher, doc=doc, chunks=chunks).consume()

    @staticmethod
    def _tx_delete_next_for_doc(tx, doc_id: str) -> None:
        cypher = """
        MATCH (:Document {doc_id:$doc_id})-[:HAS_CHUNK]->(c:Chunk)-[r:NEXT]->(:Chunk)
        DELETE r
        """
        tx.run(cypher, doc_id=doc_id).consume()

    @staticmethod
    def _tx_upsert_next_pairs(tx, pairs: List[Dict[str, Any]]) -> None:
        cypher_next = """
        UNWIND $pairs AS p
        MATCH (a:Chunk {chunk_id: p.a})
        MATCH (b:Chunk {chunk_id: p.b})
        MERGE (a)-[:NEXT]->(b)
        RETURN count(*) AS n
        """
        tx.run(cypher_next, pairs=pairs).consume()