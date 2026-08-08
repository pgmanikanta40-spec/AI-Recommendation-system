"""Core recommendation engine for the AI Recommendation System project.

The project intentionally uses only the DecodeLabs assignment algorithm:
TF-IDF vectorization plus cosine similarity for content-based filtering.
"""

from __future__ import annotations

import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendationEngine:
    """Content-based recommendation engine using TF-IDF and cosine similarity."""

    REQUIRED_COLUMNS = [
        "item_id",
        "title",
        "category",
        "type",
        "level",
        "description",
        "keywords",
    ]

    FEATURE_COLUMNS = [
        "category",
        "type",
        "level",
        "description",
        "keywords",
    ]

    def __init__(self, dataset_path: Path, screenshots_dir: Path) -> None:
        self.dataset_path = dataset_path
        self.screenshots_dir = screenshots_dir
        self.raw_data: pd.DataFrame | None = None
        self.data: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix: Any = None
        self.similarity_matrix: np.ndarray | None = None
        self.processing_time = 0.0
        self.missing_values_before = 0
        self.missing_values_after = 0
        self.duplicates_removed = 0

    def load_dataset(self) -> None:
        """Load the sample dataset from CSV."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {self.dataset_path}"
            )

        self.raw_data = pd.read_csv(self.dataset_path)
        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.raw_data.columns
        ]
        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        self.data = self.raw_data.copy()

    def handle_missing_values(self) -> None:
        """Fill missing values so the text pipeline can run safely."""
        self._ensure_data_loaded()

        self.missing_values_before = int(self.data.isna().sum().sum())

        for column in self.REQUIRED_COLUMNS:
            if column == "item_id":
                self.data[column] = self.data[column].fillna(0)
            elif column == "title":
                self.data[column] = self.data[column].fillna("Unknown Item")
            else:
                self.data[column] = self.data[column].fillna("")

        self.missing_values_after = int(self.data.isna().sum().sum())

    def clean_dataset(self) -> None:
        """Clean duplicate titles and normalize text columns."""
        self._ensure_data_loaded()

        rows_before = len(self.data)
        self.data["title_key"] = self.data["title"].astype(str).str.lower()
        self.data = self.data.drop_duplicates(subset=["title_key"])
        self.data = self.data.drop(columns=["title_key"])
        self.data = self.data.reset_index(drop=True)
        self.duplicates_removed = rows_before - len(self.data)

        for column in self.FEATURE_COLUMNS + ["title"]:
            cleaned_column = f"clean_{column}"
            self.data[cleaned_column] = (
                self.data[column]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    def extract_features(self) -> None:
        """Combine selected content columns into one feature string."""
        self._ensure_data_loaded()

        clean_columns = [f"clean_{column}" for column in self.FEATURE_COLUMNS]
        self.data["combined_features"] = (
            self.data[clean_columns].agg(" ".join, axis=1).str.strip()
        )

    def apply_tfidf_vectorization(self) -> None:
        """Apply TF-IDF vectorization to the combined feature column."""
        self._ensure_data_loaded()

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.data["combined_features"]
        )

    def calculate_cosine_similarity(self) -> None:
        """Create the cosine similarity matrix from the TF-IDF matrix."""
        self._ensure_vectorized()
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def build_engine(self) -> dict[str, Path]:
        """Generate presentation charts after the recommendation model is ready."""
        self._ensure_similarity_ready()
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        return {
            "heatmap": self.generate_similarity_heatmap(),
            "feature_chart": self.generate_tfidf_feature_chart(),
        }

    def prepare(self) -> dict[str, Path]:
        """Run the full DecodeLabs workflow from loading to model creation."""
        start_time = time.perf_counter()
        self.load_dataset()
        self.handle_missing_values()
        self.clean_dataset()
        self.extract_features()
        self.apply_tfidf_vectorization()
        self.calculate_cosine_similarity()
        chart_paths = self.build_engine()
        self.processing_time = time.perf_counter() - start_time
        return chart_paths

    def search_items(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search titles using exact, partial, and close matching."""
        self._ensure_data_loaded()

        query = query.strip().lower()
        if not query:
            return self.data.iloc[0:0]

        titles = self.data["title"].astype(str)
        lower_titles = titles.str.lower()

        partial_matches = self.data[lower_titles.str.contains(query, na=False)]
        close_titles = get_close_matches(
            query,
            lower_titles.tolist(),
            n=limit,
            cutoff=0.45,
        )
        close_matches = self.data[lower_titles.isin(close_titles)]

        results = pd.concat([partial_matches, close_matches])
        return results.drop_duplicates(subset=["title"]).head(limit)

    def recommend_items(self, user_query: str, top_n: int = 5) -> dict[str, Any]:
        """Recommend the top similar items for a title or cold-start text query."""
        self._ensure_similarity_ready()

        start_time = time.perf_counter()
        matched_index, match_message = self.find_item_index(user_query)

        if matched_index is None:
            result = self._recommend_from_text(user_query, top_n)
        else:
            result = self._recommend_from_existing_item(matched_index, top_n)
            result["matched_item"] = self.data.loc[matched_index, "title"]

        result["match_message"] = match_message
        result["execution_time"] = time.perf_counter() - start_time
        return result

    def find_item_index(self, user_query: str) -> tuple[int | None, str]:
        """Find a dataset item index from user input."""
        self._ensure_data_loaded()

        query = user_query.strip().lower()
        if not query:
            return None, "No item name was entered."

        titles = self.data["title"].astype(str)
        lower_titles = titles.str.lower()

        exact_matches = self.data[lower_titles == query]
        if not exact_matches.empty:
            index = int(exact_matches.index[0])
            return index, "Exact title match found."

        partial_matches = self.data[lower_titles.str.contains(query, na=False)]
        if not partial_matches.empty:
            index = int(partial_matches.index[0])
            title = self.data.loc[index, "title"]
            return index, f"Partial match found: {title}"

        close_titles = get_close_matches(
            query,
            lower_titles.tolist(),
            n=1,
            cutoff=0.55,
        )
        if close_titles:
            index = int(self.data[lower_titles == close_titles[0]].index[0])
            title = self.data.loc[index, "title"]
            return index, f"Closest spelling match used: {title}"

        return None, "Item was not found in the dataset."

    def get_similarity_score(
        self,
        first_item: str,
        second_item: str,
    ) -> dict[str, Any]:
        """Return the cosine similarity score between two catalog items."""
        self._ensure_similarity_ready()

        first_index, first_message = self.find_item_index(first_item)
        second_index, second_message = self.find_item_index(second_item)

        if first_index is None or second_index is None:
            return {
                "success": False,
                "first_message": first_message,
                "second_message": second_message,
                "suggestions": self.get_title_suggestions(first_item),
            }

        score = float(self.similarity_matrix[first_index][second_index])
        return {
            "success": True,
            "first_item": self.data.loc[first_index, "title"],
            "second_item": self.data.loc[second_index, "title"],
            "score": score,
            "score_percent": score * 100,
            "first_message": first_message,
            "second_message": second_message,
        }

    def get_dataset_preview(self, rows: int = 10) -> pd.DataFrame:
        """Return a readable dataset preview."""
        self._ensure_data_loaded()
        columns = ["item_id", "title", "category", "type", "level"]
        return self.data[columns].head(rows)

    def get_statistics(self) -> dict[str, Any]:
        """Return dataset and model statistics for the terminal UI."""
        self._ensure_similarity_ready()

        return {
            "dataset_size": len(self.data),
            "dataset_columns": len(self.data.columns),
            "number_of_features": len(self.FEATURE_COLUMNS),
            "feature_columns": ", ".join(self.FEATURE_COLUMNS),
            "tfidf_vocabulary_size": len(self.vectorizer.vocabulary_),
            "tfidf_matrix_size": self.tfidf_matrix.shape,
            "cosine_similarity_matrix_size": self.similarity_matrix.shape,
            "processing_time": self.processing_time,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
            "duplicates_removed": self.duplicates_removed,
        }

    def get_title_suggestions(self, query: str, limit: int = 5) -> list[str]:
        """Return close title suggestions for user-friendly errors."""
        self._ensure_data_loaded()

        lower_titles = self.data["title"].astype(str).str.lower().tolist()
        original_titles = self.data["title"].astype(str).tolist()
        title_map = dict(zip(lower_titles, original_titles))
        matches = get_close_matches(
            query.strip().lower(),
            lower_titles,
            n=limit,
            cutoff=0.35,
        )
        return [title_map[match] for match in matches]

    def generate_similarity_heatmap(self, max_items: int = 15) -> Path:
        """Save a cosine similarity heatmap using matplotlib."""
        self._ensure_similarity_ready()

        limit = min(max_items, len(self.data))
        matrix = self.similarity_matrix[:limit, :limit]
        labels = self.data["title"].head(limit).astype(str).tolist()
        short_labels = [label[:18] + "..." if len(label) > 21 else label
                        for label in labels]

        output_path = self.screenshots_dir / "cosine_similarity_heatmap.png"
        plt.figure(figsize=(12, 9))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Cosine Similarity")
        plt.xticks(range(limit), short_labels, rotation=75, ha="right")
        plt.yticks(range(limit), short_labels)
        plt.title("Cosine Similarity Heatmap")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_tfidf_feature_chart(self, top_n: int = 15) -> Path:
        """Save a chart showing the highest weighted TF-IDF features."""
        self._ensure_vectorized()

        feature_names = self.vectorizer.get_feature_names_out()
        feature_scores = np.asarray(self.tfidf_matrix.sum(axis=0)).ravel()
        top_indices = feature_scores.argsort()[-top_n:]
        top_features = feature_names[top_indices]
        top_scores = feature_scores[top_indices]

        output_path = self.screenshots_dir / "tfidf_feature_count_chart.png"
        plt.figure(figsize=(10, 7))
        plt.barh(top_features, top_scores, color="#2f80ed")
        plt.xlabel("Total TF-IDF Weight")
        plt.ylabel("Feature")
        plt.title("Top TF-IDF Feature Count Chart")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def generate_recommendation_score_chart(
        self,
        recommendations: pd.DataFrame,
        query_label: str,
    ) -> Path:
        """Save a bar chart for the latest top recommendation scores."""
        if recommendations.empty:
            raise ValueError("No recommendations available for charting.")

        safe_label = (
            query_label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        output_path = (
            self.screenshots_dir
            / f"top_recommendation_scores_{safe_label[:30]}.png"
        )

        titles = recommendations["title"].astype(str).tolist()
        short_titles = [title[:24] + "..." if len(title) > 27 else title
                        for title in titles]
        scores = recommendations["similarity_percent"].tolist()

        plt.figure(figsize=(10, 6))
        plt.barh(short_titles[::-1], scores[::-1], color="#27ae60")
        plt.xlabel("Similarity Percentage")
        plt.ylabel("Recommended Item")
        plt.title("Top Recommendation Score Chart")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return output_path

    def _recommend_from_existing_item(
        self,
        item_index: int,
        top_n: int,
    ) -> dict[str, Any]:
        similarity_scores = list(enumerate(self.similarity_matrix[item_index]))
        similarity_scores = sorted(
            similarity_scores,
            key=lambda item: item[1],
            reverse=True,
        )

        top_matches = [
            (index, score)
            for index, score in similarity_scores
            if index != item_index
        ][:top_n]
        recommendations = self._build_recommendation_frame(top_matches)

        return {
            "cold_start": False,
            "recommendations": recommendations,
            "explanation": (
                "These items were recommended because their TF-IDF content "
                "features are closest to the selected item using cosine "
                "similarity."
            ),
        }

    def _recommend_from_text(
        self,
        user_query: str,
        top_n: int,
    ) -> dict[str, Any]:
        query_text = self._clean_single_text(user_query)
        query_vector = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).ravel()

        if np.max(scores) == 0:
            return {
                "cold_start": True,
                "recommendations": pd.DataFrame(),
                "suggestions": self.get_title_suggestions(user_query),
                "explanation": (
                    "Cold start detected. The entered item is not available "
                    "in the dataset and the words entered did not match the "
                    "TF-IDF vocabulary."
                ),
            }

        top_indices = scores.argsort()[::-1][:top_n]
        top_matches = [(int(index), float(scores[index]))
                       for index in top_indices]
        recommendations = self._build_recommendation_frame(top_matches)

        return {
            "cold_start": True,
            "recommendations": recommendations,
            "suggestions": self.get_title_suggestions(user_query),
            "explanation": (
                "Cold start handled with the same TF-IDF and cosine "
                "similarity approach by comparing your text with each "
                "item's content features."
            ),
        }

    def _build_recommendation_frame(
        self,
        matches: list[tuple[int, float]],
    ) -> pd.DataFrame:
        rows = []
        for rank, (index, score) in enumerate(matches, start=1):
            item = self.data.loc[index]
            rows.append(
                {
                    "rank": rank,
                    "item_id": item["item_id"],
                    "title": item["title"],
                    "category": item["category"],
                    "type": item["type"],
                    "level": item["level"],
                    "similarity_score": round(float(score), 4),
                    "similarity_percent": round(float(score) * 100, 2),
                }
            )

        return pd.DataFrame(rows)

    def _clean_single_text(self, text: str) -> str:
        series = pd.Series([text])
        return (
            series.astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .iloc[0]
        )

    def _ensure_data_loaded(self) -> None:
        if self.data is None:
            raise RuntimeError("Dataset has not been loaded yet.")

    def _ensure_vectorized(self) -> None:
        self._ensure_data_loaded()
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF vectorization has not been applied yet.")

    def _ensure_similarity_ready(self) -> None:
        self._ensure_vectorized()
        if self.similarity_matrix is None:
            raise RuntimeError("Cosine similarity matrix has not been created.")
