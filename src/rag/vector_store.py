from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chains.contract_analysis_chain import Clause

DB_PATH = Path(__file__).parents[2] / ".contract_db"


@dataclass
class SearchResult:
    clause_id: str
    title: str | None
    text: str
    score: float


class ContractVectorStore:
    """
    계약서 조항을 ChromaDB PersistentClient 컬렉션으로 관리.

    - contract_id 단위로 컬렉션을 분리
    - 서버 재시작 후에도 인덱스 유지 (.contract_db/)
    - 멀티스레드 안전: 컬렉션 생성/삭제에 lock 사용
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._client = self._make_client(db_path)
        self._lock = threading.Lock()

    @staticmethod
    def _make_client(db_path: Path):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb가 설치되지 않았습니다. `pip install chromadb`"
            ) from exc
        db_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(db_path))

    def _collection_name(self, contract_id: str | int) -> str:
        return f"contract_{contract_id}"

    def index_clauses(self, contract_id: str | int, clauses: list[Clause]) -> None:
        """
        조항 리스트를 임베딩하여 ChromaDB에 저장.
        이미 존재하는 컬렉션은 삭제 후 재생성.
        """
        from src.rag.embedder import embed_texts

        name = self._collection_name(contract_id)
        texts = [f"{c.title or ''}\n{c.text}".strip() for c in clauses]
        ids = [c.clause_id for c in clauses]
        metadatas = [
            {"clause_id": c.clause_id, "title": c.title or "", "text": c.text}
            for c in clauses
        ]

        vectors = embed_texts(texts)

        with self._lock:
            # 기존 컬렉션 제거
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

            collection = self._client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        contract_id: str | int,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        질문과 가장 유사한 조항 top_k개 반환.
        컬렉션이 없으면 빈 리스트 반환 (폴백 허용).
        """
        from src.rag.embedder import embed_query

        name = self._collection_name(contract_id)
        try:
            collection = self._client.get_collection(name)
        except Exception:
            return []

        query_vector = embed_query(query)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["metadatas", "distances"],
        )

        search_results: list[SearchResult] = []
        metadatas_list = result.get("metadatas", [[]])[0]
        distances_list = result.get("distances", [[]])[0]

        for meta, dist in zip(metadatas_list, distances_list):
            search_results.append(SearchResult(
                clause_id=meta["clause_id"],
                title=meta.get("title") or None,
                text=meta["text"],
                score=1.0 - dist,  # cosine distance → similarity
            ))

        return search_results

    def delete(self, contract_id: str | int) -> None:
        """컬렉션 삭제 (분석 완료 후 메모리 정리)."""
        name = self._collection_name(contract_id)
        with self._lock:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

    def has_index(self, contract_id: str | int) -> bool:
        """해당 contract_id의 인덱스가 존재하는지 확인."""
        name = self._collection_name(contract_id)
        try:
            col = self._client.get_collection(name)
            return col.count() > 0
        except Exception:
            return False


# 모듈 레벨 싱글톤
_store: ContractVectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> ContractVectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ContractVectorStore()
    return _store
