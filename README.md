Web Search and Web Mining — Project 1

GitHub repository: https://github.com/YChaoWang/WebSearch_and_Mining

Project Overview

WebSearch_and_Mining is a document retrieval and evaluation system designed to retrieve and assess documents relevant to a user's query. It applies the Vector Space Model with multiple term-weighting schemes and similarity measures to search collections of English and Chinese news articles.

Main Files and Directories

main.py: Main entry point for running Tasks 1–4.

VectorSpace.py: Implements Task 1, including the Vector Space Model with different weighting schemes and similarity metrics.

Relevance_feedback.py: Implements Task 2, focusing on relevance feedback.

Evaluation.py: Implements Task 4 and evaluates the information retrieval system.

Parser.py: Handles tokenization, including NLTK-based English tokenization and Chinese word segmentation for Task 3.

EnglishNews/: English documents used in Tasks 1 and 2.

ChineseNews/: Chinese documents used in Task 3.

smaller_dataset/: Documents and queries used in Task 4.

english.stop: English stop-word list.

Project Structure

.
├── ChineseNews/
├── EnglishNews/
├── smaller_dataset/
│   ├── collections/
│   └── queries/
├── Evaluation.py
├── Parser.py
├── PorterStemmer.py
├── Relevance_feedback.py
├── VectorSpace.py
├── directory_tree.py
├── english.stop
├── main.py
├── requirements.txt
├── tfidf.py
└── util.py

Prerequisites

Python 3.6 or later

pip, the Python package manager

Setting Up a Virtual Environment

Creating a virtual environment is optional but recommended.

macOS/Linux

python -m venv venv
source venv/bin/activate

Windows

python -m venv venv
.\venv\Scripts\activate

Installing Dependencies

From the project root directory, run:

pip install -r requirements.txt

If the program reports that an additional package is missing, install that package manually using pip.

Usage

To run Tasks 1–4 with custom arguments, use:

python main.py --Eng_news_dir "./EnglishNews" --Chi_news_dir "./ChineseNews" --Eng_query "Typhoon Taiwan war" --Chi_query "資安 遊戲" --base_path "./smaller_dataset"

Alternatively, run the program with its default arguments:

python main.py

Task Outputs

Task 1: English Document Retrieval

For the English query specified by --Eng_query <EnglishQuery>, the program displays results produced by the following combinations:

Raw TF weighting with cosine similarity

Raw TF-IDF weighting with cosine similarity

Raw TF weighting with Euclidean distance

Raw TF-IDF weighting with Euclidean distance

The raw term-frequency formulation follows the method introduced in the course materials.

Task 2: Relevance Feedback

For the English query specified by --Eng_query <EnglishQuery>, the program displays retrieval results generated using relevance feedback.

Task 3: Chinese Document Retrieval

For the Chinese query specified by --Chi_query <ChineseQuery>, the program displays retrieval results after Chinese word segmentation.

Task 4: Information Retrieval Evaluation

The program reports the following evaluation metrics:

MRR@10

MAP@10

Recall@10

Conclusion

This project provides an end-to-end framework for document retrieval and evaluation using foundational information retrieval techniques. Questions, suggestions, and contributions are welcome through GitHub issues or pull requests.
