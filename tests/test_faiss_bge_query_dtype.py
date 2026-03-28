import faiss
import numpy as np


def test_faiss_search_numpy_float32() -> None:
    """FAISS IndexFlatL2.search expects float32 numpy queries (as in BGERetriever)."""
    d = 8
    index = faiss.IndexFlatL2(d)
    db = np.random.randn(3, d).astype(np.float32)
    index.add(db)
    q = np.ascontiguousarray(np.random.randn(1, d).astype(np.float32))
    _, indices = index.search(q, k=2)
    assert indices.shape == (1, 2)
    assert indices[0, 0] >= 0
