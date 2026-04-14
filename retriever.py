"""
retriever.py
Semantic retrieval over FAQ questions using a multilingual sentence-transformer.
The model encodes questions into dense vectors; cosine similarity is used at
query time.  This approach handles paraphrase, synonym, and cross-lingual
queries far better than sparse TF-IDF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import COL_QUESTION

# A compact multilingual model that handles Chinese + English well.
# Swap for a larger model (e.g. "BAAI/bge-m3") for higher accuracy.
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class FAQRetriever:
    """
    Encodes all FAQ questions into dense sentence embeddings and retrieves
    the top-k semantically closest matches for a user query.

    Design notes
    ────────────
    • Model is loaded once and cached on the instance.
    • Question embeddings are pre-computed at init time (fast at query time).
    • `model_name` is a constructor parameter so callers can swap the
      underlying model without touching search logic.
    • For large corpora, replace the numpy cosine scan with a FAISS index.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self._model = SentenceTransformer(model_name)
        self._build_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        questions = self.df[COL_QUESTION].tolist()
        # encode returns (n, d) float32 numpy array
        self._embeddings: np.ndarray = self._model.encode(
            questions,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit vectors → dot == cosine
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Returns a list of result dicts ranked by semantic similarity.
        Each dict contains all FAQ columns plus 'score' (0-1) and 'rank'.
        """
        if not query.strip():
            return []

        q_vec = self._model.encode(
            [query],
            normalize_embeddings=True,
        )  # shape (1, d)

        scores = cosine_similarity(q_vec, self._embeddings).flatten()

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx].to_dict()
            row["score"] = float(scores[idx])
            row["rank"] = rank
            results.append(row)
        return results
