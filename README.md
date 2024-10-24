# WebSearch_and_WebMining Project1
Github repo👉<https://github.com/YChaoWang/WebSearch_and_WebMining.git>

## 專案描述
**WebSearch_and_WebMining** 是一個文檔搜索和評估系統，旨在根據用戶查詢方便地檢索和評估相關文檔。該專案利用向量空間模型和各種相似性度量方法，在新聞文章集合中執行搜索。

## 主要檔案
- **main.py**: 主要執行檔案。
- **VectorSpace.py**: 實現Task1，包含Vector Space Model with Different Weighting Schemes & Similarity Metrics。
- **Relevance_feedback.py**: 實現Task2，專注於Relevance Feedback機制。
- **Evaluation.py**: 實現Task4，評估信息檢索（IR）系統。
- **Parser.py**: 處理分詞，包括 NLTK 分詞和Task3的中文分詞。

### 先決條件
- Python 3.6 或更高版本
- pip（Python 套件管理器）

### 創建虛擬環境（可選但推薦）
要為此專案創建虛擬環境，請運行以下命令：

```bash
# 創建名為 'venv' 的虛擬環境
python -m venv venv

# 啟用虛擬環境
source venv/bin/activate  # 在 macOS/Linux 上
.\venv\Scripts\activate  # 在 Windows 上
````


### 安裝依賴
專案根目錄下有一個 `requirements.txt` 文件。要安裝所需的套件，請運行：

````bash
pip install -r requirements.txt
````
備註：如遇到需要另行安裝的，再麻煩手動安裝，不好意思


## 使用方法
要執行Task1至4，請使用以下命令：

````bash
python main.py --Eng_news_dir "./EnglishNews" --Chi_news_dir "./ChineseNews" --Eng_query "Typhoon Taiwan war" --Chi_query "資安 遊戲" --base_path "./smaller_dataset"
````


或者，您可以運行：

````bash
python main.py
````


這將使用默認參數執行程式。

### 任務輸出
- **Task1**: 您將看到對於指定的英語查詢 `--Eng_query <EnglishQuery>`的結果：
  - TF Weighting（Course PPT 中討論的Raw TF）+ Cosine Similarity
  - TF-IDF Weighting（Course PPT 中討論的Raw TF）+ Cosine Similarity
  - TF Weighting（Course PPT 中討論的Raw TF）+ Euclidean Distance
  - TF-IDF Weighting（Course PPT 中討論的Raw TF）+ Euclidean Distance

- **Task2**: 對於指定的英語查詢 `--Eng_query <EnglishQuery>`，將顯示Relevance Feedback的結果。

- **Task3**: 對於指定的中文查詢 `--Chi_query <ChineseQuery>`，將顯示結果。

- **Task4**: 您將看到評估指標，包括：
  - MRR@10
  - MAP@10
  - Recall@10

## 結論
此專案提供了一個全面的文檔檢索和評估框架，利用信息檢索中的先進技術。如有任何問題或貢獻，請隨時聯繫或提交拉取請求。
