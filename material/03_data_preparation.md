# 第三章：準備我們的原料 - K線資料 (Hands-on Part 1)

量化分析的基礎是數據。數據的品質直接決定了分析結果的可靠性。"Garbage in, garbage out." (垃圾進，垃圾出)。

### 3.1 認識 K 線 (Candlestick)

K 線是金融市場的標準語言，它在一個小小的圖示中包含了豐富的資訊。我們使用的是「分K」，代表每一分鐘市場的價格快照。

一根 K 棒包含五個關鍵資訊 **OHLCV**:
-   **Open (開盤價):** 該分鐘的第一筆成交價。
-   **High (最高價):** 該分鐘內的最高成交價。
-   **Low (最低價):** 該分鐘內的最低成交價。
-   **Close (收盤價):** 該分鐘的最後一筆成交價。
-   **Volume (成交量):** 該分鐘內的總成交量。

### 3.2 實戰：資料讀取與清理

我們的第一個實作任務：將原始的 `.txt` 檔，變成乾淨、可供分析的 `DataFrame`。

**目標：**
1.  讀取文字檔。
2.  處理時間欄位，將其設為 `DataFrame` 的索引 (Index)。
3.  檢查資料基本狀況。
4.  繪製初步的走勢圖。

**步驟與程式碼範例 (在 Jupyter Notebook 中執行):**

1.  **導入函式庫並讀取資料**
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt

    # 設定圖表樣式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 檔案路徑 (請根據你的實際路徑修改)
    file_path = '../data/TXF1_Minute_2020-01-01_2025-06-16.txt'
    
    # 讀取 CSV 檔案
    # 假設欄位名稱為 'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'
    # 如果你的檔案沒有表頭，需要手動指定 names
    df = pd.read_csv(file_path)
    
    print("Original data head:")
    print(df.head())
    ```

2.  **時間格式處理 (最重要的一步！)**
    時間序列分析的基礎，就是擁有一個正確的 `datetime` 索引。
    ```python
    # 將 'Date' 和 'Time' 欄位合併成一個字串
    datetime_str = df['Date'] + ' ' + df['Time']

    # 將字串轉換為 pandas 的 datetime 物件
    df['Datetime'] = pd.to_datetime(datetime_str, format='%Y/%m/%d %H:%M')

    # 將新的 'Datetime' 欄位設為 DataFrame 的索引
    df = df.set_index('Datetime')

    # 移除舊的、不再需要的 'Date' 和 'Time' 欄位
    df = df.drop(['Date', 'Time'], axis=1)

    print("\nProcessed data head with Datetime index:")
    print(df.head())
    ```

3.  **初步探索**
    ```python
    # 檢查欄位資料型態和是否有缺失值
    print("\nData information:")
    df.info()

    # 取得描述性統計
    print("\nStatistical summary:")
    print(df.describe())
    ```

4.  **視覺化**
    讓我們畫出收盤價的走勢，對數據有一個宏觀的認識。
    ```python
    # 繪製收盤價走勢圖
    df['Close'].plot(figsize=(15, 7), title='TXF1 Close Price (2020-2025)', lw=1)
    plt.ylabel('Price')
    plt.show()
    ```

完成這一步後，我們就有了一個乾淨、結構化、隨時可以進行分析的數據基礎了。 