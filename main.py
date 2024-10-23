import glob
import os
import argparse
from VectorSpace import VectorSpace
from Evaluation import evaluate_ir_system


def main():
    parser = argparse.ArgumentParser(
        description="Document Search and Evaluation System"
    )

    # 設定預設的新聞資料夾
    parser.add_argument(
        "--news_dir",
        type=str,
        default="./EnglishNews",  # 預設的新聞資料夾路徑
        help="Directory containing the news files.",
    )

    # 設定預設查詢字串
    parser.add_argument(
        "--Eng_query",
        type=str,
        default="Typhoon Taiwan war",  # 預設的查詢字串
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

    # 獲取新聞檔案
    news_files = sorted(
        glob.glob(os.path.join(args.news_dir, "News*.txt")),
        key=lambda x: int(x.split("News")[-1].split(".")[0]),
    )
    print("Task1: VSM with Different Weighting Schemes & Similarity Metrics")

    print(f"Found {len(news_files)} documents")

    # 讀取檔案
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

    # 建立 VectorSpace 物件並建構向量空間
    print("Building vector space...")
    vectorSpace = VectorSpace(documents)

    # 查詢結果
    query = args.Eng_query
    # Perform search using Raw TF + Cosine similarity
    print(f"\nSearch results for {query} using Raw TF and Cosine similarity:")
    vectorSpace.search(
        query.split(), method="cosine", weighting="tf", file_paths=file_paths
    )

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

    # Perform search using TF-IDF + Euclidean distance
    print(f"\nSearch results for {query} using TF-IDF and Euclidean distance:")
    vectorSpace.search(
        query.split(), method="euclidean", weighting="tf-idf", file_paths=file_paths
    )

    # 執行評估
    print("\nRunning full evaluation...")
    queries_path = os.path.join(args.base_path, "queries")
    collections_path = os.path.join(args.base_path, "collections")
    rel_file = os.path.join(args.base_path, "rel.tsv")

    evaluate_ir_system(queries_path, collections_path, rel_file)


if __name__ == "__main__":
    main()
