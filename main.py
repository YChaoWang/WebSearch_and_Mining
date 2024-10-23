import glob
import os
import argparse
from VectorSpace import VectorSpace
from Evaluation import evaluate_ir_system
from Relevance_feedback import pseudo_feedback  # Import the pseudo_feedback function


def load_documents(news_dir):
    """Load news documents from the specified directory."""
    news_files = sorted(
        glob.glob(os.path.join(news_dir, "News*.txt")),
        key=lambda x: int(x.split("News")[-1].split(".")[0]),
    )
    print(f"Found {len(news_files)} documents")

    documents = []
    file_paths = []

    for file_path in news_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
                if content:
                    documents.append(content)
                    file_paths.append(file_path)
                else:
                    print(f"Warning: Empty document {file_path} skipped.")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return documents, file_paths


def perform_vsm_search(vectorSpace, query, file_paths, language):
    # Perform search using Raw TF + Cosine similarity
    print(f"\nSearch results for {query} using Raw TF and Cosine similarity:")
    vectorSpace.search(
        query.split(), method="cosine", weighting="tf", file_paths=file_paths
    )

    if language == "english":
        # Perform search using Raw TF + Euclidean distance
        print(f"\nSearch results for {query} using Raw TF and Euclidean distance:")
        vectorSpace.search(
            query.split(), method="euclidean", weighting="tf", file_paths=file_paths
        )

    # Perform search using TF-IDF + Cosine similarity
    print(f"\nSearch results for {query} using TF-IDF and Cosine similarity:")
    vectorSpace.search(
        query.split(), method="cosine", weighting="tf-idf", file_paths=file_paths
    )

    if language == "english":
        # Perform search using TF-IDF + Euclidean distance
        print(f"\nSearch results for {query} using TF-IDF and Euclidean distance:")
        vectorSpace.search(
            query.split(), method="euclidean", weighting="tf-idf", file_paths=file_paths
        )


def task2(vectorSpace, original_query, documents, file_paths):
    """Task 2: Perform Pseudo Feedback and re-rank documents."""
    print(f"\nTask 2: Performing Pseudo Feedback for query: {original_query}")
    pseudo_feedback(vectorSpace, original_query, documents, file_paths)

    # if re_ranked_results:

    #     print("\nNewsID Score")
    #     for index, score in re_ranked_results:
    #         if index < len(file_paths):
    #             file_name = os.path.basename(file_paths[index])
    #             print(f"{file_name}  {score:.7f}")
    # else:
    #     print("No results found after Pseudo Feedback.")


def main():
    parser = argparse.ArgumentParser(
        description="Document Search and Evaluation System"
    )

    # 設定預設的新聞資料夾
    parser.add_argument(
        "--eng_news_dir",
        type=str,
        default="./EnglishNews",  # 預設的英文新聞資料夾路徑
        help="Directory containing the English news files.",
    )

    parser.add_argument(
        "--chi_news_dir",
        type=str,
        default="./ChineseNews",  # 預設的中文新聞資料夾路徑
        help="Directory containing the Chinese news files.",
    )

    # 設定預設查詢字串
    parser.add_argument(
        "--Eng_query",
        type=str,
        default="Typhoon Taiwan war",  # 預設的查詢字串
        help="Query for document search.",
    )

    parser.add_argument(
        "--Chi_query",
        type=str,
        default="資安 遊戲",
        help="Query for document search.",
    )

    # 設定預設的評估檔案路徑
    parser.add_argument(
        "--base_path",
        type=str,
        default="./smaller_dataset",  # 預設的評估檔案路徑
        help="Base path for evaluation files (queries, collections, rel.tsv).",
    )

    args = parser.parse_args()

    # Load documents
    print("Task 1: VSM with Different Weighting Schemes & Similarity Metrics")
    eng_documents, eng_file_paths = load_documents(args.eng_news_dir)

    # 建立 VectorSpace 物件並建構向量空間
    print("Building Task 1 vector space...")
    engVectorSpace = VectorSpace(eng_documents, language="english")

    #
    if args.Eng_query:
        query = args.Eng_query
        print(f"\nEnglish query: {query}")
        perform_vsm_search(engVectorSpace, query, eng_file_paths, "english")

        # Task 2: Perform Pseudo Feedback
        task2(engVectorSpace, query, eng_documents, eng_file_paths)

    print(
        "Task 3: VSM with Different Scheme & Similarity Metrics in Chinese and English"
    )
    chi_documents, chi_file_paths = load_documents(args.chi_news_dir)

    print("Building Task 3 vector space...")
    chiVectorSpace = VectorSpace(chi_documents, language="chinese")

    # 中文查詢
    if args.Chi_query:
        query = args.Chi_query
        print(f"\nChinese query: {query}")
        perform_vsm_search(chiVectorSpace, query, chi_file_paths, "chinese")

    # 執行評估
    print("\nRunning full evaluation...")
    queries_path = os.path.join(args.base_path, "queries")
    collections_path = os.path.join(args.base_path, "collections")
    rel_file = os.path.join(args.base_path, "rel.tsv")

    evaluate_ir_system(queries_path, collections_path, rel_file)  # Task 4


if __name__ == "__main__":
    main()
