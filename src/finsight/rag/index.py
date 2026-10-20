"""Persistent Chroma index with explicit embeddings and atomic generation publication."""

import hashlib
import math
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from finsight.errors import DisclosureError, EmbeddingError
from finsight.rag.embeddings import Embedder
from finsight.rag.models import DisclosureHit, DisclosureRef, DocumentChunk, IndexResult


class DisclosureIndex:
    def __init__(self, path: str | Path, embedder: Embedder):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=str(self.path / "vectors"), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="finsight-disclosures-v1",
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedder.model, "dimensions": embedder.dimensions},
        )
        meta = self.collection.metadata or {}
        if (
            meta.get("embedding_model") != embedder.model
            or meta.get("dimensions") != embedder.dimensions
        ):
            raise DisclosureError(
                "기존 색인의 임베딩 모델/차원이 다릅니다. 별도 DB 경로를 사용하세요."
            )
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS active_documents (
                document_key TEXT PRIMARY KEY, generation TEXT NOT NULL,
                receipt_no TEXT NOT NULL, chunks INTEGER NOT NULL, source_sha256 TEXT NOT NULL,
                indexed_at TEXT NOT NULL)""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path / "manifest.sqlite3", timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def active(self, key: str) -> dict[str, Any] | None:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM active_documents WHERE document_key = ?", (key,)
            ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _where(reference: DisclosureRef, generation: str) -> dict[str, Any]:
        return {
            "$and": [
                {"corp_code": {"$eq": reference.company.corp_code}},
                {"business_year": {"$eq": reference.business_year}},
                {"report_type": {"$eq": reference.report_type}},
                {"receipt_no": {"$eq": reference.receipt_no}},
                {"generation": {"$eq": generation}},
            ]
        }

    def publish(
        self, reference: DisclosureRef, digest: str, chunks: list[DocumentChunk]
    ) -> IndexResult:
        if not chunks or len({chunk.id for chunk in chunks}) != len(chunks):
            raise DisclosureError("빈 문서 또는 중복 조각 ID는 색인할 수 없습니다.")
        for chunk in chunks:
            if (
                chunk.metadata.get("document_key") != reference.key
                or chunk.metadata.get("corp_code") != reference.company.corp_code
                or chunk.metadata.get("business_year") != reference.business_year
                or chunk.metadata.get("report_type") != reference.report_type
                or chunk.metadata.get("receipt_no") != reference.receipt_no
                or chunk.metadata.get("source_sha256") != digest
                or chunk.metadata.get("corp_name") != reference.company.corp_name
                or chunk.metadata.get("source") != reference.source
                or chunk.metadata.get("member") != f"{reference.receipt_no}.xml"
            ):
                raise DisclosureError("색인 조각의 출처 메타데이터가 문서와 다릅니다.")
        identity = ":".join(
            [digest, self.embedder.model, str(self.embedder.dimensions)]
            + [chunk.id for chunk in chunks]
        )
        generation = hashlib.sha256(identity.encode()).hexdigest()
        existing = self.collection.get(where=self._where(reference, generation), include=[])["ids"]
        existing_ids = set(existing)
        missing = [chunk for chunk in chunks if chunk.id not in existing_ids]
        embedded = 0
        # Each successful batch is reusable after an interrupted indexing attempt.
        for start in range(0, len(missing), 24):
            batch = missing[start : start + 24]
            texts = [
                f"{reference.company.corp_name} {reference.business_year} 사업보고서\n"
                f"{chunk.metadata['section']}\n{chunk.text}"
                for chunk in batch
            ]
            vectors = self.embedder.embed(texts)
            if len(vectors) != len(batch):
                raise EmbeddingError("임베딩 벡터 개수가 문서 조각 수와 다릅니다.")
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[{**chunk.metadata, "generation": generation} for chunk in batch],
                embeddings=vectors,
            )
            embedded += len(batch)
        # Only publish after all records exist; queries always filter by this generation.
        with self._db() as db:
            db.execute(
                """INSERT INTO active_documents VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_key) DO UPDATE SET generation=excluded.generation,
                receipt_no=excluded.receipt_no, chunks=excluded.chunks,
                source_sha256=excluded.source_sha256, indexed_at=excluded.indexed_at""",
                (
                    reference.key,
                    generation,
                    reference.receipt_no,
                    len(chunks),
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return IndexResult(reference, generation, len(chunks), embedded, not missing, digest)

    def candidates(
        self, reference: DisclosureRef, query: str, *, limit: int = 100
    ) -> list[DisclosureHit]:
        active = self.active(reference.key)
        if active is None or active["receipt_no"] != reference.receipt_no:
            return []
        vector = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[vector],
            where={
                "$and": self._where(reference, active["generation"])["$and"]
                + [{"kind": {"$eq": "paragraph"}}]
            },
            n_results=min(limit, active["chunks"]),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            if (
                not math.isfinite(distance)
                or meta["corp_code"] != reference.company.corp_code
                or meta["business_year"] != reference.business_year
                or meta["report_type"] != reference.report_type
                or meta["receipt_no"] != reference.receipt_no
                or meta["generation"] != active["generation"]
                or meta["corp_name"] != reference.company.corp_name
                or meta["source"] != reference.source
                or meta["member"] != f"{reference.receipt_no}.xml"
            ):
                raise DisclosureError("검색 결과의 출처/조건 검증에 실패했습니다.")
            hits.append(DisclosureHit(chunk_id, text, float(distance), meta))
        return hits
