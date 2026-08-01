from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.chains.contract_analysis_chain import Clause
    from src.rag.hybrid_search import BM25Index

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
        self._bm25_cache: dict[str, "BM25Index"] = {}

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
        조항 리스트를 부모-자식 청크로 분할 후 자식 청크 단위로 임베딩하여 ChromaDB에 저장.
        이미 존재하는 컬렉션(및 BM25 캐시)은 삭제 후 재생성.
        """
        from src.rag.chunking import split_into_sub_chunks
        from src.rag.embedder import embed_texts

        name = self._collection_name(contract_id)

        sub_chunks = [sc for clause in clauses for sc in split_into_sub_chunks(clause)]
        if not sub_chunks:
            return

        ids = [sc.sub_chunk_id for sc in sub_chunks]
        texts = [sc.text for sc in sub_chunks]
        metadatas = [
            {
                "parent_clause_id": sc.parent_clause_id,
                "parent_title": sc.parent_title or "",
                "parent_text": sc.parent_text,
                "chunk_order": sc.order,
            }
            for sc in sub_chunks
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
            self._bm25_cache.pop(str(contract_id), None)

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
        Dense 검색만 수행 (자식 청크 단위 → parent 조항으로 그룹핑 후 top_k 반환).
        하이브리드 검색을 못 쓰는 상황(rank_bm25 미설치 등)을 위한 순수 dense 폴백 경로.
        컬렉션이 없으면 빈 리스트 반환.
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
            n_results=min(top_k * 4, collection.count()),
            include=["metadatas", "distances"],
        )

        metadatas_list = result.get("metadatas", [[]])[0]
        distances_list = result.get("distances", [[]])[0]

        seen_parents: set[str] = set()
        search_results: list[SearchResult] = []
        for meta, dist in zip(metadatas_list, distances_list):
            parent_id = meta.get("parent_clause_id")
            if not parent_id or parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            search_results.append(SearchResult(
                clause_id=parent_id,
                title=meta.get("parent_title") or None,
                text=meta.get("parent_text", ""),
                score=1.0 - dist,  # cosine distance → similarity
            ))
            if len(search_results) >= top_k:
                break

        return search_results

    def hybrid_search(
        self,
        contract_id: str | int,
        query: str,
        top_k: int = 5,
        dense_k: int = 20,
        sparse_k: int = 20,
        rrf_k: int = 10,
    ) -> list[SearchResult]:
        """
        Dense(ChromaDB) + Sparse(BM25) 자식 청크 검색을 RRF로 재랭킹하여
        parent 조항 단위 top_k를 반환한다.

        - Dense: 의미 유사도 기반. 표현이 달라도(동의어/우회 표현) 관련 조항을 찾음
        - Sparse(BM25): "제12조", "위약금" 같은 법률 핵심 키워드의 정확 매칭 재현율을 보완
        - rank_bm25 미설치/BM25 인덱스 구축 실패 시 dense 결과만으로 자동 폴백 (RRF는 빈 리스트를
          병합해도 안전하게 동작)
        - rrf_k=10: RRF 원 논문의 k=60은 후보군이 수백~수천 개인 웹 검색 기준값이라, 계약서 1건당
          조항이 수십 개 수준인 우리 도메인에 그대로 쓰면 순위 차이가 거의 희석되어 버림(1위와 3위의
          점수차가 노이즈 수준으로 작아짐). 후보군 크기에 맞춰 k를 낮춰 상위 순위 신호를 더 크게 반영한다.
        """
        from src.rag.embedder import embed_query
        from src.rag.hybrid_search import reciprocal_rank_fusion

        name = self._collection_name(contract_id)
        try:
            collection = self._client.get_collection(name)
        except Exception:
            return []

        # ── Dense 검색 (자식 청크 단위) ──
        query_vector = embed_query(query)
        dense_result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(dense_k, collection.count()),
            include=["metadatas", "distances"],
        )
        dense_ids: list[str] = dense_result.get("ids", [[]])[0]
        dense_metas = dict(zip(dense_ids, dense_result.get("metadatas", [[]])[0]))
        dense_dists = dict(zip(dense_ids, dense_result.get("distances", [[]])[0]))

        # ── Sparse 검색 (BM25, 자식 청크 단위) ──
        try:
            bm25 = self._get_bm25_index(contract_id)
            sparse_ids = bm25.search(query, top_k=sparse_k) if bm25 else []
        except Exception:
            sparse_ids = []

        if not dense_ids and not sparse_ids:
            return []

        # ── RRF 병합 ──
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=rrf_k)

        # BM25 전용 히트(dense 결과에 없던 sub-chunk)는 메타데이터를 Chroma에서 보강 조회
        missing = [doc_id for doc_id, _ in fused if doc_id not in dense_metas]
        if missing:
            fetched = collection.get(ids=missing, include=["metadatas"])
            dense_metas.update(dict(zip(fetched.get("ids", []), fetched.get("metadatas", []))))

        # ── 자식 청크 → parent 조항 그룹핑, top_k 부모만 채택 ──
        seen_parents: set[str] = set()
        results: list[SearchResult] = []
        for doc_id, fused_score in fused:
            meta = dense_metas.get(doc_id)
            if not meta:
                continue
            parent_id = meta.get("parent_clause_id")
            if not parent_id or parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            similarity = 1.0 - dense_dists[doc_id] if doc_id in dense_dists else fused_score
            results.append(SearchResult(
                clause_id=parent_id,
                title=meta.get("parent_title") or None,
                text=meta.get("parent_text", ""),
                score=similarity,
            ))
            if len(results) >= top_k:
                break

        return results

    def _get_bm25_index(self, contract_id: str | int) -> "BM25Index | None":
        """계약서 단위 BM25 인덱스를 캐시에서 반환, 없으면 Chroma에 저장된 자식 청크로 구축."""
        from src.rag.hybrid_search import BM25Document, BM25Index

        key = str(contract_id)
        with self._lock:
            cached = self._bm25_cache.get(key)
        if cached is not None:
            return cached

        name = self._collection_name(contract_id)
        try:
            collection = self._client.get_collection(name)
            data = collection.get(include=["documents"])
        except Exception:
            return None

        ids = data.get("ids", [])
        docs = data.get("documents", [])
        if not ids:
            return None

        index = BM25Index([BM25Document(doc_id=i, text=d) for i, d in zip(ids, docs)])
        with self._lock:
            self._bm25_cache[key] = index
        return index

    def delete(self, contract_id: str | int) -> None:
        """컬렉션 및 BM25 캐시 삭제 (분석 완료 후 메모리 정리)."""
        name = self._collection_name(contract_id)
        with self._lock:
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
            self._bm25_cache.pop(str(contract_id), None)

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
