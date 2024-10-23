# WebSearch_and_WebMining

## 專案描述
WebSearch_and_WebMining 是一個文檔搜索和評估系統，旨在根據用戶查詢方便地檢索和評估相關文檔。該專案利用向量空間模型和各種相似性度量方法，在新聞文章集合中執行搜索。

## 主要檔案
- main.py 主要執行檔案
- VectorSpace.py: Task1 VSM with Different Weighting Schemes & Similarity Metrics
- Relevance_feedback: Task2
- Evaluation.py: Task4 Evaluation IR System

- Parser.py: 進行分詞、nltk.tokenize 以及Task3 中文分詞等

### 先決條件
- Python 3.6 或更高版本
- pip（Python 套件管理器）



### 創建虛擬環境（可選但推薦）
```bash
python -m venv venv
source venv/bin/activate  # 在 macOS/Linux 上
.\venv\Scripts\activate  # 在 Windows 上
```


### 安裝依賴
專案根目錄下有一個 `requirements.txt` 文件。

安裝所需的套件：
```bash
pip install -r requirements.txt
```


## 使用方法
要執行Task1~4，請使用如以下命令：
```bash
python main.py --news_dir "./EnglishNews" --Eng_query "Typhoon Taiwan war" --Chi_query "資安 遊戲" --base_path "./smaller_dataset"
```
或是
```bash
python main.py 
```
會執行default參數

Task1 會看到以下:
TF Weighting (Raw TF in course PPT) + Cosine Similarity
TF-IDF Weighting (Raw TF in course PPT) + Cosine Similarity
TF Weighting (Raw TF in course PPT) + Euclidean Distance
TF-IDF Weighting (Raw TF in course PPT) + Euclidean Distance

Task2 會看到--Eng_query <EnglishQuery> 的結果

Task3 會看到--Chi_query <ChineseQuery> 的結果

Task4 會看到以下:
MRR@10
MAP@10
Recall@10





