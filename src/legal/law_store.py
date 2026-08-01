"""
법령 전용 벡터 저장소.

- 영구 파일 저장 (clair-ai/.law_db/) — 법령은 자주 바뀌지 않으므로 재사용
- 법령명 단위로 컬렉션 분리
- 서버 시작 시 정적 데이터로 자동 초기화
"""
from __future__ import annotations

import threading
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).parents[2] / ".law_db"


@dataclass
class LawSearchResult:
    law_name: str
    article_no: str
    article_title: str
    content: str
    score: float


class LawVectorStore:
    """
    법령 조항 영구 벡터 저장소.
    ChromaDB PersistentClient 사용 — 서버 재시작 후에도 인덱스 유지.
    """

    COLLECTION_NAME = "korean_law"

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._client = self._make_client(db_path)
        self._lock = threading.Lock()
        self._initialized = False

    @staticmethod
    def _make_client(db_path: Path):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb가 필요합니다: pip install chromadb") from exc
        db_path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(db_path))

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def initialize(self, force: bool = False) -> int:
        """
        정적 법령 데이터로 벡터 DB 초기화.
        이미 데이터가 있으면 건너뜀 (force=True면 재구축).
        반환값: 인덱싱된 조항 수
        """
        from src.legal.fetcher import fetch_articles
        from src.rag.embedder import embed_texts

        with self._lock:
            collection = self._get_or_create_collection()

            if not force and collection.count() > 0:
                self._initialized = True
                return collection.count()

            if force:
                self._client.delete_collection(self.COLLECTION_NAME)
                collection = self._client.create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )

            articles = fetch_articles()  # API 키 있으면 law.go.kr, 없으면 정적 데이터
            if not articles:
                return 0

            # 동일 법령+조문번호 중복 제거 (API가 같은 조문을 여러 번 반환할 수 있음)
            seen_ids: set[str] = set()
            unique_articles = []
            for a in articles:
                uid = f"{a.law_name}_{a.article_no}"
                if uid not in seen_ids:
                    seen_ids.add(uid)
                    unique_articles.append(a)
            articles = unique_articles

            texts = [f"{a.law_name} {a.article_no} {a.article_title}\n{a.content}" for a in articles]
            ids = [f"{a.law_name}_{a.article_no}" for a in articles]
            metadatas = [
                {
                    "law_name": a.law_name,
                    "article_no": a.article_no,
                    "article_title": a.article_title,
                    "content": a.content,
                    "contract_types": ",".join(a.contract_types),
                }
                for a in articles
            ]

            # Gemini 임베딩 API: 1배치 최대 100개 제한 → 청크로 나눠서 처리
            BATCH_SIZE = 100
            all_vectors = []
            for i in range(0, len(texts), BATCH_SIZE):
                all_vectors.extend(embed_texts(texts[i:i + BATCH_SIZE]))

            collection.add(ids=ids, embeddings=all_vectors, documents=texts, metadatas=metadatas)
            self._initialized = True
            return len(articles)

    def search(
        self,
        query: str,
        contract_type: str | None = None,
        top_k: int = 3,
    ) -> list[LawSearchResult]:
        """
        계약 조항 텍스트와 유사한 법령 조항 검색.
        contract_type 지정 시 해당 계약 유형 법령만 검색.
        """
        from src.rag.embedder import embed_query

        collection = self._get_or_create_collection()
        if collection.count() == 0:
            return []

        # contract_type 필터는 Python 쪽에서 처리 (ChromaDB where 미사용)
        fetch_k = collection.count()
        query_vector = embed_query(query)
        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(fetch_k, collection.count()),
            include=["metadatas", "distances"],
        )

        results: list[LawSearchResult] = []
        for meta, dist in zip(
            result.get("metadatas", [[]])[0],
            result.get("distances", [[]])[0],
        ):
            # contract_type 필터 — 저장된 값이 comma-separated string
            if contract_type and contract_type not in meta.get("contract_types", ""):
                continue
            results.append(LawSearchResult(
                law_name=meta["law_name"],
                article_no=meta["article_no"],
                article_title=meta["article_title"],
                content=meta["content"],
                score=1.0 - dist,
            ))
            if len(results) >= top_k:
                break

        return results

    def is_ready(self) -> bool:
        try:
            return self._get_or_create_collection().count() > 0
        except Exception:
            return False


# ── 싱글톤 ────────────────────────────────────────────────────────────────────

_law_store: LawVectorStore | None = None
_law_store_lock = threading.Lock()


def get_law_store() -> LawVectorStore:
    global _law_store
    if _law_store is None:
        with _law_store_lock:
            if _law_store is None:
                _law_store = LawVectorStore()
    return _law_store


def ensure_law_db_initialized() -> int:
    """
    서버 시작 시 호출 — 법령 DB가 비어있으면 자동 초기화.
    Gemini API 키 없으면 건너뜀 (임베딩 불가).
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")

    if not os.environ.get("GEMINI_API_KEY", ""):
        return 0
    try:
        return get_law_store().initialize()
    except Exception:
        return 0
