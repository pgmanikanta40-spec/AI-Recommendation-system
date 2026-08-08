"""Menu-driven terminal application for Project 3 - AI Recommendation System."""

from __future__ import annotations

import time
from pathlib import Path

from recommendation_engine import RecommendationEngine
from utils import (
    Color,
    ask_continue,
    color_text,
    dataframe_to_rows,
    exit_application,
    format_table,
    loading_step,
    print_banner,
    print_error,
    print_info,
    print_menu,
    print_section,
    print_success,
    print_warning,
    prompt_integer,
    prompt_non_empty,
)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "sample_recommendation_items.csv"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"


def initialize_application() -> RecommendationEngine:
    """Run the complete startup workflow required by the assignment."""
    print_banner()

    engine = RecommendationEngine(DATASET_PATH, SCREENSHOTS_DIR)
    start_time = time.perf_counter()

    loading_step("Loading Dataset", engine.load_dataset)
    loading_step("Cleaning Data", _make_cleaning_action(engine))
    loading_step("Generating TF-IDF Matrix", _make_tfidf_action(engine))
    loading_step("Calculating Cosine Similarity", engine.calculate_cosine_similarity)
    chart_paths = loading_step("Building Recommendation Engine",
                               engine.build_engine)

    engine.processing_time = time.perf_counter() - start_time

    print_success("\nReady!")
    print(color_text("================================================", Color.CYAN))
    print_startup_summary(engine)
    print_info(f"Heatmap saved to: {chart_paths['heatmap']}")
    print_info(f"TF-IDF chart saved to: {chart_paths['feature_chart']}")
    return engine


def _make_cleaning_action(engine: RecommendationEngine):
    """Return a startup action for missing values, cleaning, and extraction."""

    def action() -> None:
        engine.handle_missing_values()
        engine.clean_dataset()
        engine.extract_features()

    return action


def _make_tfidf_action(engine: RecommendationEngine):
    """Return a startup action for TF-IDF vectorization."""

    def action() -> None:
        engine.apply_tfidf_vectorization()

    return action


def print_startup_summary(engine: RecommendationEngine) -> None:
    """Display important dataset and model information after startup."""
    stats = engine.get_statistics()
    rows = [
        {"Metric": "Dataset Size", "Value": stats["dataset_size"]},
        {"Metric": "Number of Features", "Value": stats["number_of_features"]},
        {
            "Metric": "TF-IDF Vocabulary Size",
            "Value": stats["tfidf_vocabulary_size"],
        },
        {
            "Metric": "Cosine Similarity Matrix Size",
            "Value": stats["cosine_similarity_matrix_size"],
        },
        {
            "Metric": "Processing Time",
            "Value": f"{stats['processing_time']:.4f} seconds",
        },
    ]
    print(format_table(rows, ["Metric", "Value"]))


def view_dataset(engine: RecommendationEngine) -> None:
    """Display a preview of the dataset."""
    print_section("Dataset Preview")
    total_rows = engine.get_statistics()["dataset_size"]
    rows = prompt_integer(
        f"How many rows do you want to view? (1-{total_rows}, default 10): ",
        minimum=1,
        maximum=total_rows,
        default=min(10, total_rows),
    )
    preview = engine.get_dataset_preview(rows)
    print(format_table(dataframe_to_rows(preview), preview.columns.tolist()))


def show_dataset_statistics(engine: RecommendationEngine) -> None:
    """Display model and dataset statistics."""
    print_section("Dataset Statistics")
    stats = engine.get_statistics()
    rows = [
        {"Metric": "Dataset Size", "Value": stats["dataset_size"]},
        {"Metric": "Dataset Columns", "Value": stats["dataset_columns"]},
        {"Metric": "Number of Features", "Value": stats["number_of_features"]},
        {"Metric": "Feature Columns", "Value": stats["feature_columns"]},
        {
            "Metric": "TF-IDF Vocabulary Size",
            "Value": stats["tfidf_vocabulary_size"],
        },
        {"Metric": "TF-IDF Matrix Size", "Value": stats["tfidf_matrix_size"]},
        {
            "Metric": "Cosine Similarity Matrix Size",
            "Value": stats["cosine_similarity_matrix_size"],
        },
        {
            "Metric": "Processing Time",
            "Value": f"{stats['processing_time']:.4f} seconds",
        },
        {
            "Metric": "Missing Values Before",
            "Value": stats["missing_values_before"],
        },
        {
            "Metric": "Missing Values After",
            "Value": stats["missing_values_after"],
        },
        {"Metric": "Duplicates Removed", "Value": stats["duplicates_removed"]},
    ]
    print(format_table(rows, ["Metric", "Value"]))


def search_item(engine: RecommendationEngine) -> None:
    """Search for catalog items."""
    print_section("Search Item")
    query = prompt_non_empty("Enter item title or keyword: ")
    results = engine.search_items(query)

    if results.empty:
        print_warning("No matching item found.")
        suggestions = engine.get_title_suggestions(query)
        if suggestions:
            print_info("Did you mean one of these?")
            for title in suggestions:
                print(f"- {title}")
        return

    display_columns = ["item_id", "title", "category", "type", "level"]
    print(format_table(dataframe_to_rows(results[display_columns]),
                       display_columns))


def recommend_items(engine: RecommendationEngine) -> None:
    """Generate and display top five recommendations."""
    print_section("Recommend Items")
    query = prompt_non_empty("Enter an item title or content keywords: ")

    result = engine.recommend_items(query, top_n=5)
    recommendations = result["recommendations"]

    print_info(result["match_message"])
    if result["cold_start"]:
        print_warning("Cold start problem detected and handled.")

    if recommendations.empty:
        print_error("No recommendation could be generated for this input.")
        suggestions = result.get("suggestions", [])
        if suggestions:
            print_info("Try one of these dataset items:")
            for title in suggestions:
                print(f"- {title}")
        return

    display_columns = [
        "rank",
        "title",
        "category",
        "level",
        "similarity_percent",
    ]
    print(format_table(dataframe_to_rows(recommendations[display_columns]),
                       display_columns))

    print_section("Recommendation Explanation")
    print(result["explanation"])
    print(
        "Higher percentages mean the item has more similar words and content "
        "features in the TF-IDF vector space."
    )
    print_success(
        f"Recommendation generation time: {result['execution_time']:.4f} "
        "seconds"
    )

    chart_path = engine.generate_recommendation_score_chart(
        recommendations,
        result.get("matched_item", query),
    )
    print_info(f"Top recommendation score chart saved to: {chart_path}")


def view_similarity_score(engine: RecommendationEngine) -> None:
    """Display cosine similarity between two items."""
    print_section("View Similarity Score")
    first_item = prompt_non_empty("Enter first item title: ")
    second_item = prompt_non_empty("Enter second item title: ")
    result = engine.get_similarity_score(first_item, second_item)

    if not result["success"]:
        print_error("Could not calculate similarity score.")
        print_warning(result["first_message"])
        print_warning(result["second_message"])
        if result["suggestions"]:
            print_info("Possible matches:")
            for title in result["suggestions"]:
                print(f"- {title}")
        return

    rows = [
        {"Metric": "First Item", "Value": result["first_item"]},
        {"Metric": "Second Item", "Value": result["second_item"]},
        {
            "Metric": "Cosine Similarity",
            "Value": f"{result['score']:.4f}",
        },
        {
            "Metric": "Similarity Percentage",
            "Value": f"{result['score_percent']:.2f}%",
        },
    ]
    print(format_table(rows, ["Metric", "Value"]))


def show_pipeline() -> None:
    """Show the exact assignment workflow."""
    print_section("Recommendation Pipeline")
    steps = [
        "1. Load Dataset",
        "2. Display Dataset Information",
        "3. Handle Missing Values",
        "4. Clean Dataset",
        "5. Feature Extraction",
        "6. Apply TF-IDF Vectorization",
        "7. Generate TF-IDF Matrix",
        "8. Calculate Cosine Similarity Matrix",
        "9. Build Recommendation Engine",
        "10. Accept User Input",
        "11. Recommend Top 5 Most Similar Items",
        "12. Display Recommendation Scores",
        "13. Handle Cold Start Problem",
        "14. Allow Continuous Recommendations until Exit",
    ]
    for step in steps:
        print(color_text(step, Color.WHITE))


def run_application() -> None:
    """Run the terminal menu until the user exits."""
    try:
        engine = initialize_application()

        actions = {
            1: view_dataset,
            2: show_dataset_statistics,
            3: search_item,
            4: recommend_items,
            5: view_similarity_score,
        }

        while True:
            print_menu()
            choice = prompt_integer("Enter your choice (1-7): ", 1, 7)

            if choice == 6:
                try:
                    show_pipeline()
                except Exception as error:
                    print_error(str(error))
            elif choice == 7:
                exit_application()
            else:
                action = actions[choice]
                try:
                    action(engine)
                except Exception as error:
                    print_error(str(error))
                    print_warning("Please try another menu option.")

            if not ask_continue():
                exit_application()

    except KeyboardInterrupt:
        print_warning("\nApplication interrupted by user.")
    except Exception as error:
        print_error(str(error))
        print_warning("The application stopped safely without crashing.")


if __name__ == "__main__":
    run_application()
