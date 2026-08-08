# AI Recommendation System

Project 3 - AI Recommendation System for the DecodeLabs Artificial
Intelligence Training Kit.

This project is a professional, terminal-based content recommendation system.
It strictly follows the DecodeLabs concepts of feature extraction, TF-IDF
vectorization, cosine similarity, and content-based filtering.

## Project Overview

The application recommends the top 5 most similar items from a sample dataset.
It uses item metadata such as category, type, level, description, and keywords
to build a TF-IDF matrix. Cosine similarity is then calculated to identify
which items are most similar to the user's selected item or content query.

No deep learning, cloud API, database, web framework, LLM, or external AI
service is used.

## Objectives

- Load and inspect a dataset.
- Handle missing values and clean text data.
- Extract useful item content features.
- Convert text features into numerical vectors using TF-IDF.
- Calculate a cosine similarity matrix.
- Recommend the top 5 most similar items.
- Display recommendation scores and explanation.
- Handle spelling mistakes and cold start inputs.
- Provide a professional terminal interface for live demonstration.

## Features

- Professional ASCII AI banner.
- Loading animation during startup.
- ANSI colored terminal output.
- Clean menu-driven interface.
- Dataset preview and statistics.
- Item search with close spelling matching.
- Top 5 content-based recommendations.
- Similarity percentages for every recommendation.
- Recommendation explanation after every prediction.
- Cosine similarity score comparison between two items.
- Cold start handling with TF-IDF text matching.
- Continuous recommendations until the user exits.
- Matplotlib visualizations saved in the `screenshots/` folder.

## Recommendation Workflow

1. Load Dataset
2. Display Dataset Information
3. Handle Missing Values
4. Clean Dataset
5. Feature Extraction
6. Apply TF-IDF Vectorization
7. Generate TF-IDF Matrix
8. Calculate Cosine Similarity Matrix
9. Build Recommendation Engine
10. Accept User Input
11. Recommend Top 5 Most Similar Items
12. Display Recommendation Scores
13. Handle Cold Start Problem
14. Allow continuous recommendations until Exit

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn

## Installation

Clone or download the project folder, then install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python main.py
```

## Folder Structure

```text
AI_Recommendation_System/
|
|-- main.py
|-- recommendation_engine.py
|-- utils.py
|-- dataset/
|   |-- sample_recommendation_items.csv
|-- screenshots/
|   |-- cosine_similarity_heatmap.png
|   |-- tfidf_feature_count_chart.png
|   |-- top_recommendation_scores_<item>.png
|-- requirements.txt
|-- README.md
```

## Screenshots

The application automatically saves the following visualization files:

- `screenshots/cosine_similarity_heatmap.png`
- `screenshots/tfidf_feature_count_chart.png`
- `screenshots/top_recommendation_scores_<item>.png`

These files can be used in GitHub, LinkedIn, reports, and DecodeLabs
submission material.

## Sample Menu

```text
1. View Dataset
2. Dataset Statistics
3. Search Item
4. Recommend Items
5. View Similarity Score
6. Show Recommendation Pipeline
7. Exit
```

## Future Improvements

- Add a larger CSV dataset with more item categories.
- Add more visual summaries using matplotlib.
- Export recommendation results to CSV.
- Add more unit tests for data cleaning and recommendations.
- Improve terminal table formatting for very small screens.

## Author

Created for DecodeLabs evaluation by:

**Your Name**
