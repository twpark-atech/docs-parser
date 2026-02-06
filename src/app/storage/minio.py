# ==============================================================================
# 목적 : MinIO 유틸
# 최초 작업자 : (AI솔루션/박태원)
# 최초 작업일 : 2026-01-15
# AI 활용 여부 :
# ==============================================================================

import io
from dataclasses import dataclass
from typing import Dict, Any, Optional

from minio import Minio


@dataclass(frozen=True)
class MinIOConfig:
    """MinIO(S3 호환) 접근 설정 클래스.
    MinIO Python SDK(Minio 클라이언트) 초기화에 필요한 접속 정보와 기본 버킷을 보관합니다.

    Attributes:
        endpoint: MinIO endpoint.
        access_key: 액세스 키.
        secret_key: 비밀번호 키.
        bucket: 기본 사용할 버킷명.
        secure: HTTPS 사용 여부.
    """
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool = False


class MinIOWriter:
    """MinIO에 object를 업로드하는 Writer 클래스.
    
    ensure_bucket(): 버킷이 없으면 생성.
    build_*_key(): 프로젝트 규칙에 따른 object key 생성.
    upload_*(): bytes 또는 파일을 지정 key로 업로드하고 메타(etag/size)를 반환
    stat(): 오브젝트 메타데이터 조회
    """
    def __init__(self, cfg: MinIOConfig):
        """Writer를 초기화합니다.

        Args:
            cfg: MinIOConfig 설정 객체.
        """
        self._cfg = cfg
        self._client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )


    def ensure_bucket(self) -> None:
        """기본 버킷(cfg.bucket)이 존재하도록 보장합니다.
        
        이미 존재하면 아무 것도 하지 않습니다.
        존재하지 않으면 make_bucket으로 생성합니다.
        """
        if self._client.bucket_exists(self._cfg.bucket):
            return
        self._client.make_bucket(self._cfg.bucket)


    def build_object_key(self, sha256_hex: str, filename: str = "original") -> str:
        """sha256 기반의 content-addressable object key를 생성합니다.
        
        동일 콘텐츠(sha256)가 같은 prefix에 모이도록 하여 디렉토리 분산 효과를 얻습니다.
        키 포맷: sha256/{h[:2]}/{h[2:4]}/{h}/{filename}

        Args:
            sha256_hex: 파일/바이트의 SHA-256 hex 문자열.
            filename: 오브젝트 파일명.

        Returns:
            생성된 object key 문자열.
        """
        h = sha256_hex.lower()
        return f"sha256/{h[:2]}/{h[2:4]}/{h}/{filename}"
    
    def build_pdf_key(self, doc_id: str, filename: str = "original.pdf") -> str:
        """문서(doc_id) 기준 PDF 저장 경로 key를 생성합니다.
        
        키 포맷: pdf/{doc_id}/{filename}

        Args:
            doc_id: 문서 식별자.
            filename: 저장 파일명.

        Returns:
            생성된 object key 문자열.
        """
        return f"pdf/{doc_id}/{filename}"
    

    def build_crop_image_key(self, doc_id: str, image_id: str, ext: str = "png") -> str:
        """문서(doc_id) 내 crop 이미지 저장 경로 key를 생성합니다.
        
        키 포맷: pdf/{doc_id}/images/{image_id}.{ext}

        Args:
            doc_id: 문서 식별자.
            image_id: 이미지 식별자.
            ext: 확장자.

        Returns:
            생성된 object key 문자열.
        """
        return f"pdf/{doc_id}/images/{image_id}.{ext}"
    

    def upload_bytes_to_key(
        self,
        data: bytes,
        *,
        object_key: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """바이트 데이터를 지정 object_key로 업로드합니다.
        
        업로드 전 ensure_bucket()으로 버킷 존재를 보장합니다.
        put_object를 사용하며 content_type이 없으면 "application/octet_stream"을 사용합니다.

        Args:
            data: 업로드할 바이트 데이터.
            object_key: 저장할 object_key.
            content_type: MIME type.

        Returns:
            업로드 결과 메타 dict
            - bucket: 대상 버킷명.
            - key: object key.
            - etag: 업로드 결과 etag.
            - size_bytes: 업로드한 바이트 크기.
        """
        self.ensure_bucket()
        stream = io.BytesIO(data)
        size = len(data)
        result = self._client.put_object(
            bucket_name=self._cfg.bucket,
            object_name=object_key,
            data=stream,
            length=size,
            content_type=content_type or "application/octet_stream",
        )
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(result, "etag", None),
            "size_bytes": size,
        }


    def upload_file_to_key(
        self,
        file_path: str,
        *,
        object_key: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """로컬 파일을 지정 object_key로 업로드합니다.
        
        업로드 전 ensure_bucket()으로 버킷 존재를 보장합니다.
        fput_object로 업로드 후, stat_object로 size를 조회해 반환합니다.

        Args:
            file_path: 업로드할 로컬 파일 경로.
            object_key: 저장할 object key.
            content_type: MIME type.

        Returns:
            업로드 결과 메타 dict
            - bucket: 대상 버킷명.
            - key: object key.
            - etag: 업로드 결과 etag.
            - size_bytes: stat_object로 조회한 크기.
        """
        self.ensure_bucket()
        result = self._client.fput_object(
            bucket_name=self._cfg.bucket,
            object_name=object_key,
            file_path=file_path,
            content_type=content_type or "application/octet_stream",
        )
        stat = self._client.stat_object(self._cfg.bucket, object_key)
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(result, "etag", None),
            "size_bytes": getattr(stat, "size", None),
        }
 

    def upload_bytes(
        self,
        data: bytes,
        sha256_hex: str,
        filename: str,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """sha256 기반 object key 규칙으로 bytes를 업로드합니다.
        
        build_object_key(sha256_hex, filename)로 key를 생성한 뒤 upload_bytes_to_key를 호출합니다.

        Args:
            data: 업로드할 바이트 데이터.
            sha256_hex: 데이터의 sha256 hex 문자열.
            filename: object key에 포함될 파일명.
            content_type: MIME type.

        Returns:
            upload_bytes_to_key와 동일한 결과 dict.
        """
        self.ensure_bucket()
        object_key = self.build_object_key(sha256_hex, filename=filename)
        return self.upload_bytes_to_key(data, object_key=object_key, content_type=content_type)
    

    def upload_file(
        self,
        file_path: str,
        sha256_hex: str,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """sha256 기반 object key 규칙으로 파일을 업로드합니다.

        filename이 없으면 file_path의 basename을 사용합니다.
        build_object_key로 key 생성 후 upload_file_to_key를 호출합니다.

        Args:
            file_path: 업로드할 로컬 파일 경로.
            sha256_hex: 파일의 sha256 hex 문자열.
            filename: object key에 포함될 파일명.
            content_type: MIME type.

        Returns:
            upload_file_to_key와 동일한 결과 dict.
        """
        self.ensure_bucket()
        name = filename or file_path.split("/")[-1]
        object_key = self.build_object_key(sha256_hex, filename=name)
        return self.upload_file_to_key(file_path, object_key=object_key, content_type=content_type)
    

    def stat(self, object_key: str) -> Dict[str, Any]:
        """오브젝트 메타 데이터를 조회합니다.
        
        Args:
            object_key: 조회할 object key.

        Returns:
            메타 dict
            - bucket, key, etag, size_bytes, content_type, last_modified.
        """
        st = self._client.stat_object(self._cfg.bucket, object_key)
        return {
            "bucket": self._cfg.bucket,
            "key": object_key,
            "etag": getattr(st, "etag", None),
            "size_bytes": getattr(st, "size", None),
            "content_type": getattr(st, "content_type", None),
            "last_modified": str(getattr(st, "last_modified", "")),
        }


class MinIOReader:
    """MinIO에서 오브젝트를 다운로드하는 Reader 클래스.
    get_object로 스트림을 받아 전체 바이트로 읽어 반환합니다.
    반환 후 resp.close() 및 res.release_conn()을 호출해 연결을 정리합니다.\
    """
    def __init__(self, cfg: MinIOConfig):
        """Reader를 초기화합니다.
        
        Args:
            cfg: MinIOConfig 설정 객체.
        """
        self._cfg = cfg
        self._client = Minio(
            endpoint=cfg.endpoint,
            access_key=cfg.access_key,
            secret_key=cfg.secret_key,
            secure=cfg.secure,
        )

    def download_bytes(self, *, bucket: str, key: str) -> bytes:
        """지정 bucket/key의 오브젝트를 바이트로 다운로드합니다.
        
        Args:
            bucket: 대상 버킷명.
            key: object key.

        Returns:
            object 전체 bytes.
        """
        resp = self._client.get_object(bucket, key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()