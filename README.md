# Stock-Predict
開始之前：腦力激盪
曾經寫過相關的股價分析模型，像是普通的LSTM做regression和漲跌Classification分析，或是使用NLP去抓取網路財經新聞文章，使用本人實驗室曾經做出的股價相關的情緒辭典，但效果都非常不好。所以這次與同屆同學討論出相關方法，這位同學曾經用Reinforcement Learning的”Rainbow”模型寫出股票相關論文，雖說他是使用RL做出碩論，但是他跟我建議絕不要使用強化學習且使用NLP的前處理方式也過於繁雜，需要耗費時間很長。
最後他只跟我說使用最相關的數據做前處理，製造最能當作判斷未來的資料特徵比較有用。給予的建議是產生KD MACD RSI MA等等的資料性質特徵，最後拿此特徵做樹狀決策的LightGBM模型下去試試看。
以上為做之前的想法。
了解TaLib
Ta-lib是金融軟件中應用廣泛的專門用來計算技術指標的開源庫，涵蓋了200多種市場常見的技術指標運算。它支持java,C,C++,Perl,Python等多種語言。Ricequant的java平台上也同樣引入了這個庫。在各種語言中，Ta-lib的python wrapper是最簡潔優美的，語法幾乎不需要解釋就能完全看懂。
# colab install packages
# install yfinance
! pip install yfinance
# install talib
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip install Ta-Lib
在最一開始作業要求中就有一個Talib套件的下載動作，由於我的2080ti server是使用Windows，所以果然只好在Colab上下載會是比較好的，所以TaLib我是運行在Colab上操作。
了解技術指標
RSI指標：
相對強弱指標（Relative Strength Index，RSI），顧名思義，是以股價在一段期間內多空方的相對強弱情形，來判斷未來走勢的一項指標，其公式如下：
RSI＝n日漲幅平均值÷（n日漲幅平均值＋n日跌幅平均值）×100%
指標運用上是以50%作為分水嶺，RSI大於50%，代表期間內多方力道相對較強，後市可望看漲，投資人應跟隨趨勢作多；反之，RSI小於50%，則代表期間內空方力道較強，未來有較大的下跌壓力，應該賣出持股。此外，如果RSI高於80%或低於20%（也可以使用70%與30%），則代表期間內上漲或下跌的相對比重過高，有過熱或過冷的跡象，股價可能出現回檔或反彈（但未必代表趨勢反轉），此時投資人應該適當減碼，或調整交易部位。
KD指標：
這項指標必須分別計算出K值（快線）和D值（慢線），然後再使用快慢線交錯的原理，區別黃金交叉與死亡交叉，作為進出場的依據。
在計算K值及D值之前，我們必須先計算一個名叫「未成熟隨機值（RSV）」的數值，其公式如下：
RSV＝（今日收盤價－最近n天最低價）÷（最近n天最高價－最近n天最低價）×100
計算RSV時，一般都以９日作為期間長度參數（投資人當然也可以依標的屬性加以調整），RSV則代表目前股價在計算期間內，最高與最低價格間的相對位置，接著可以分別計算K值與D值，公式如下：
當日K值＝前日K值×（2/3）＋當日RSV×（1/3）
當日D值＝前日D值×（2/3）＋當日K值×（1/3）
MA指標：
移動平均，分為簡單移動平均SMA, 和加權移動平均WMA；
移動平均（英語：Moving Average，MA），又稱“移動平均線”簡稱均線，是技術分析中一種分析時間序列數據的工具。最常見的是利用股價、回報或交易量等變量計算出移動平均。移動平均可撫平短期波動，反映出長期趨勢或週期。原本的意思是移動平均，由於我們將其製作成線形，所以一般稱之為移動平均線，簡稱均線。它是將某一段時間的收盤價之和除以該週期。比如日線MA5指5天內的收盤價除以5。
移動平均線常用線有5天、10天、30天、60天、120天和240天的指標。其中，5天和10天的短期移動平均線。是短線操作的參照指標，稱做日均線指標；30天和60天的是中期均線指標，稱做季均線指標；120天、240天的是長期均線指標，稱做年均線指標。對移動平均線的考查一般從幾個方面進行。
移動平均線按時間週期長短分為：短期移動平均線，中期移動平均線，長期移動平均線；按計算方法分為：算術移動平均​​線，加權移動平均線，指數平滑移動平均線（EMA）。
MACD指標：
平滑異同移動平均線指標（Moving Average Convergence Divergence，MACD）在1970年代由美國人阿佩爾（Gerald Appel）提出，是一項歷史悠久且經常在交易中被使用的技術分析工具，其原理是利用快慢線的交錯來判斷股價走勢的轉折，但計算方式較為複雜，投資人雖然可以直接將其買賣訊號運用在交易上，在計算MACD時，首先必須先計算長、短天期的「指數移動平均（Exponential Moving Average，EMA）」。EMA也是計算一定天期的平均價格，只是在計算上較近的日期會給予較高的權重；反之，較遠的日期則給予較低的權重。接著將長短天期EMA的相減，所得出差額就是「差離值（DIF）」，代表短期EMA偏離長期EMA的情形。
然後，我們將DIF線再作一次指數移動平均，就會得出「訊號線（DEM）」，也稱為「MACD線」，由於DIF線只經過一次指數移動平均平滑處理，對股價變動的反應較為迅速，因此我們把它視為「快線」，而經過兩次指數移動平均平滑處理的DEM線，對股價變動反應較為遲緩，我們將其視為「慢線」。
當快線由下向上突破慢線時，即為黃金交叉，代表股價後市看漲；反之，當快線由上而下跌破慢線時，則為死亡交叉，代表後市看跌。
最後，我們把快慢線相減，其差額就是我們在MACD指標圖形中所看見的柱狀線（MACD BAR/OSC），柱狀線由負轉正及由正轉負，與前述的黃金交叉、死亡交叉代表同樣的意涵，而柱狀線正負向長度越長，則代表近期股價偏向多方或空方的力道越強。在使用上必須注意的是，MACD柱狀線在盤整走勢中，可能在零軸附近上下徘徊，此時指標不具太大參考價值，此外，在漲多回檔的格局中，MACD柱狀線亦可能出現短暫跌破零軸又再次翻正的干擾訊號（跌深反彈時亦同），此時就應搭配其他基本面或技術面指標作一同參考，方能避免失真。
Data Preprocessing
import talib
from talib import abstract
data = data.astype('float')
## 技術面資料
# 改成 TA-Lib 可以辨識的欄位名稱
data.rename(columns={"Open": "open", "High": "high", "Low":"low", "Close":"close", "Volume":"volume"} , inplace=True)
Close = [float(x) for x in data['close']]
# make technical Analysis values
data['MA5'] = talib.MA(data['close'], timeperiod=5)
data['MA10'] = talib.MA(data['close'], timeperiod=10)
data['MA20'] = talib.MA(data['close'], timeperiod=20)
data['k'], data['d'] = talib.STOCH(data['high'], data['low'], data['close'])
data['MACD'],data['MACDsignal'],data['MACDhist'] = talib.MACD(np.array(Close),fastperiod=6, slowperiod=12, signalperiod=9)
data['RSI5'] = talib.RSI(data['close'], timeperiod=5)
data['RSI10'] = talib.RSI(data['close'], timeperiod=10)
data['RSI20'] = talib.RSI(data['close'], timeperiod=20)
# 預測股價
data['y']=data['close'].shift(-5)
# 五日後漲標記 1，反之標記 0
data['y'] = np.where(data.close.shift(-5) > data.close, 1, 0)
使用5天、10天、20天的”Close”計算MA
talib.STOCH計算出每日K、D值
data[‘MACD’]、data[‘MACDsignal’]、data[‘MACDhist’]分別是
macd = 6天EMA — 12天EMA
macdsignal = 9天MACD的EMA
macdhist = MACD — MACD signal
使用5天、10天、20天的”Close”計算RSI
最後發現這個ipy檔寫著五日後的漲跌，就腦動大開，從來沒想過試著預測五天後漲跌，覺得是個不錯的想法，用五天之間差來做買進賣出，感覺可以嘗試，所以五天Close差距當Label。
Image for post
TaLib資料前處理
data.plot(y = 'close')
Image for post
可看出Apple的股價一直在上升。
stock['close_diff'] = None
for i in range(1,len(stock['close'])):
    stock['close_diff'][i] = stock['close'][i] - stock['close'][i-1]
stock['diff_rate'] = (stock['close'].shift(1)/stock['close'])-1
不使用絕對數值 ”open”’、 ”close”，使用相對數值。
計算 ”Close_diff” 今天 "Close" 比前一天 ”Close” 的數值差，當作一個資料。
計算 ”diff_rate” 今天 “Close” 比前一天 ”Close” 的數值差的比例，當作一個資料。
stock['year'] = None
stock['month'] = None
stock['day'] = None
for i in range(len(stock['Date'])):
    stock['year'][i] = stock['Date'][i].split('-')[0]
    stock['month'][i] = stock['Date'][i].split('-')[1]
    stock['day'][i] = stock['Date'][i].split('-')[2]
覺得月份季節年份都會影響股價，所以使用詞彙分割將他們個別成為資料特徵。
最後只使用的特徵：
['volume', 'MA5', 'MA10', 'MA20', 'k', 'd', 'MACD', 'MACDsignal',
'MACDhist', 'RSI5', 'RSI10', 'RSI20', 'close_diff', 'year', 'month',
'day', 'diff_rate']
第N次腦動大開：
分割年分當作train 、 Valid 、 Test Data，認為上市很久，且又是突飛猛進的股票，照理來說古早以前的資料早已跟現在的資料序列有很大差異，所以我認為不應該用太久遠的資料當作訓練資料。
train_date_min = 20060101
train_date_max = 20151231
val_date_min = 20160101
val_date_max = 201801231
test_date_min = 20160101
test_date_max = int(stock['Date'][9914])
因為過去2008、2009有金融海嘯，我認為有機會預測2020肺炎疫情造成的股災。
2006–2015：train
2016–2018：Valid
2019–Now：Test
def idx_range(df,minn,maxx):
    index = []
    for i in range(len(df)):
        if minn <= int(df[i]) & int(df[i]) <= maxx:
            index.append(i)
    
    return index[0], index[-1]
train_data_idx = idx_range(stock['Date'],train_date_min,train_date_max)
val_data_idx = idx_range(stock['Date'],val_date_min,val_date_max)
test_data_idx = idx_range(stock['Date'],test_date_min,test_date_max)
train_data = stock_data[['volume', 'MA5', 'MA10', 'MA20', 'k', 'd', 'MACD', 'MACDsignal', 'MACDhist', 'RSI5', 'RSI10',
       'RSI20', 'close_diff', 'year', 'month', 'day', 'diff_rate']][train_data_idx[0]:train_data_idx[1]+1].values
train_y = stock_y[train_data_idx[0]:train_data_idx[1]+1].values
val_data = stock_data[['volume', 'MA5', 'MA10', 'MA20', 'k', 'd', 'MACD', 'MACDsignal', 'MACDhist', 'RSI5', 'RSI10',
       'RSI20', 'close_diff', 'year', 'month', 'day', 'diff_rate']][val_data_idx[0]:val_data_idx[1]+1].values
val_y = stock_y[val_data_idx[0]:val_data_idx[1]+1].values
test_data = stock_data[['volume', 'MA5', 'MA10', 'MA20', 'k', 'd', 'MACD', 'MACDsignal', 'MACDhist', 'RSI5', 'RSI10',
       'RSI20', 'close_diff', 'year', 'month', 'day', 'diff_rate']][test_data_idx[0]:test_data_idx[1]+1-5].values
test_y = stock_y[test_data_idx[0]:test_data_idx[1]+1-5].values
做了一連串的前處理，總算將資料train valid test分割完。
LightGBM
LightGBM是什麼？
LightGBM是一個梯度提升框架，使用基於樹的學習算法。
和其他的基於樹的算法有什麼不同？
LightGBM樹的生長方式是垂直方向的，其他的算法都是水平方向的，也就是說Light GBM生長的是樹的葉子，其他的算法生長的是樹的層次。LightGBM選擇具有最大誤差的樹葉進行生長，當生長同樣的樹葉，生長葉子的算法可以比基於層的算法減少更多的loss。
以下優點：
更快的訓練效率
低記憶體使用
更高的準確率
支援並行化學習
可處理大規模資料
特徵並行
Image for post
投票並行
Image for post
資料並行
Image for post

""" First import all the required libraries """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, make_scorer
引入套件
""" LightGBM Implementation and Results"""
# We need to convert our training data into LightGBM dataset format
d_train = lgb.Dataset(train_data, label=train_y)
# Setting parameters for training
# objective set to binary for binary classification problem
# boosting_type set to gbdt for gradient boosting
# binary_logloss as metric for binary classification predictions
# other parameters randomly selected and subject to change for optimization
params = {'boosting_type': 'gbdt',
          'learning_rate': 0.003,
          'max_depth': 10,
          'metric': 'binary_logloss',
          'min_data': 50,
          'num_leaves': 10,
          'objective': 'binary',
          'sub_feature': 0.5}
# fit the clf_LGBM on training data with 100 training iterations
clf_LGBM = lgb.train(params, d_train, 100)
# make predictions with test data
y_pred = clf_LGBM.predict(val_data)
# sinec the output is a list of probabilities, below we have converted the probabilities
# to binary prediction with threshold set at 0.5
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:
        y_pred[i] = 1
    else:  
        y_pred[i]=0
# evaluate predictions with accuracy metric
clf_LGBM_accuracy = accuracy_score(val_y, y_pred)
# evaluate predictions with F1-score metric
clf_LGBM_f1 = f1_score(val_y, y_pred)
print("LightGBM Classifier [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf_LGBM_accuracy, clf_LGBM_f1))
隨意調參數，這裡可能不是很正確，沒使用準確計算Threshold當作判斷結果，直接計算0.5當Threshold，最後結果：
Image for post
參數亂調準確率極低
""" Optimization of LightGBM """
# Choose LGBM Classifier as the algorithm for optimization with GridSearch
clf_LGBM2 = lgb.LGBMClassifier(boosting_type = 'gbdt', metric = 'binary_logloss', 
                               min_data = 50, objective = 'binary', sub_feature = 0.5)
# Create a dictionary for the parameters
gridParams = {'learning_rate': [0.0001, 0.0003, 0.0005, 0.001],'n_estimators': [75, 100, 125],
             'num_leaves': [15, 16, 17],'colsample_bytree' : [0.58, 0.60, 0.62],'subsample' : [0.4, 0.5, 0.7]}
# Choose the time series cross-validator
tscv = TimeSeriesSplit(n_splits=3)
# Create the GridSearch object
grid = GridSearchCV(clf_LGBM2, gridParams, verbose=1, cv= tscv)
# Fit the grid search object to the data to compute the optimal model
grid_fit_LGBM = grid.fit(train_data, train_y)
# Return the optimal model after fitting the data
best_clf_LGBM = grid_fit_LGBM.best_estimator_
# Make predictions with the optimal model
best_predictions_LGBM = best_clf_LGBM.predict(val_data)
# Get the accuracy and F1_score of the optimized model
clf_LGBM_optimized_accuracy = accuracy_score(val_y, best_predictions_LGBM)
clf_LGBM_optimized_f1 = f1_score(val_y, best_predictions_LGBM)
print("LGBM Classifier Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf_LGBM_optimized_accuracy, clf_LGBM_optimized_f1))
print(grid_fit_LGBM.best_params_)
print(grid_fit_LGBM.best_score_)
優化LightGBM，使用GridSearch找出最好參數。
Image for post
計算結果在valid data上的Accuracy Score：0.6093，f1-score：0.7572
這裡準確數值差異原因在於上漲下跌的Label幾乎是二分之一，所以在f1-score上分數可能較高。
Image for post
上漲數據還是偏高，六成好像很正常
best_predictions_LGBM = best_clf_LGBM.predict(test_data)
# Get the accuracy and F1_score of the optimized model
clf_LGBM_optimized_accuracy = accuracy_score(test_y, best_predictions_LGBM)
clf_LGBM_optimized_f1 = f1_score(test_y, best_predictions_LGBM)
print("LGBM Classifier Optimized [Accuracy score: {:.4f}, f1-score: {:.4f}]".format(clf_LGBM_optimized_accuracy, clf_LGBM_optimized_f1))
print(grid_fit_LGBM.best_params_)
print(grid_fit_LGBM.best_score_)
Image for post
計算結果在test data上的Accuracy Score：0.6121，f1-score：0.7594
在test資料上準確率較高一些，但還是差不多。
Image for post
做特徵重要程度，發現跟月份關係很重要，還有MA關聯大，可能跟出手機的月份關係有關聯也說不定。
LSTM
LSTM(Long short-term memory)，主要由四個Component組成: Input Gate、Output Gate、Memory Cell以及Forget Gate。
Input Gate: 當將feature輸入時，input gate會去控制是否將這次的值輸入
Memory Cell: 將計算出的值儲存起來，以利下個階段拿出來使用
Output Gate: 控制是否將這次計算出來的值output
Forget Gate: 是否將Memory清掉(format)，有點restart的概念。
其中，“是否”這件事情，是可以透過神經網路進行學習。
接下來我們可以更近一步去看數學表示方式：
Image for post
LSTM Model type
LSTM有許多種變化，如下圖：
One to One 、 One to many 、 Many to one 或者 Many to many
Image for post
而我們要使用many to one的方式，因為是5天預測1天。
fit資料進去，因為神經網路跟lightgbm不一樣，資料特徵得先做標準化，因為特徵區間相差非常大，會導致在使用梯度下降法尋求最佳解時，需要很迭代多次才可以收斂。
#定義正規化函式
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm
Image for post
def train_windows(df, df1, ref_day=5, predict_day=1):
    X_train = []
    Y_train = []
    for i in range(df.shape[0]-ref_day):
        X_train.append(np.array(df.iloc[i:i+ref_day,:]))
        Y_train.append(df1.iloc[i])
    return np.array(X_train), np.array(Y_train)
X, Y=train_windows(norm_data,up_down,5,1)
將資料排序好
train_data_idx_new = train_data_idx[0]-5, train_data_idx[1]-5
val_data_idx_new = val_data_idx[0]-5, val_data_idx[1]-5
test_data_idx_new = test_data_idx[0]-5, test_data_idx[1]-5
train_data = X[train_data_idx_new[0]:train_data_idx_new[1]+1]
train_y = Y[train_data_idx_new[0]:train_data_idx_new[1]+1]
val_data = X[val_data_idx_new[0]:val_data_idx_new[1]+1]
val_y = Y[val_data_idx_new[0]:val_data_idx_new[1]+1]
test_data = X[test_data_idx_new[0]:test_data_idx_new[1]+1]
test_y = Y[test_data_idx_new[0]:test_data_idx_new[1]+1]
將資料train valid test分割完。
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC' #A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth =True
set_session(tf.Session(config=config))
#======================================================================
config.gpu_options.per_process_gpu_memory_fraction = 0.8
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
regressor = Sequential()
regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (train_data.shape[1], train_data.shape[2])))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 256))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1, activation='sigmoid'))
regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
filepath = 'weights-best_.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
regressor.fit(train_data, train_y, verbose=2, callbacks=[checkpoint], validation_data = (val_data,val_y), epochs = 30, batch_size = 32)
然後這裡發生奇怪的事情，我覺得很多不合理，我認為很不合理，但我已經找了好久資料哪裡出問題，卻一直找不到，我認為一定有用到未來資料在裡面，查片對應的feature、label，卻不知道哪裡有錯。
準確率高達八成。
Image for post
regressor.evaluate(test_data,test_y)
Image for post
test data也有八成。實在太不合理了。
有發現問題的同仁拜託跟我說。
為了搞清楚資料對應問題，確定資料對應沒錯，舉例：
每五天對上一個label，此label是用第一天跟第六天的close做比較。
我這麼做準確率是八成以上…
****又或者真的這麼準，依照趨勢狀況，第一天第六天比較可以用LSTM看這五天的趨勢也說不定。****
Image for post
最後心得：
其實玩過股票分析一陣子，但其實對於金融方向領域知識還是偏向需要更熟練，純粹只能依照自己的邏輯跟有限的資料數據餵進自己已知的模型中嘗試看看準確率，這次發現從KD、MA等等數值嘗試可以知道還有更多資料生成特徵在模型權重重要性是偏高的，只是還是不能理解自己做的LSTM模型錯在哪裡，可能還得花些許時間了解。
當然我還有很多腦動大開的地方，像是使用APPLE公司上下游企業的資料嘗試餵進去，或是使用NLP模型concate在一起做分析等等的，雖然經驗告訴自己用NLP concate效果可能會變差，但還是想嘗試看看。
最後，由於時間有限，我沒辦法做更多很抱歉。有問題我會再更改，感謝。
參考來源：
https://www.wealth.com.tw/home/articles/20244
https://www.finlab.tw/Python-%E7%B0%A1%E5%96%AE158%E7%A8%AE%E6%8A%80%E8%A1%93%E6%8C%87%E6%A8%99%E8%A8%88%E7%AE%97/
https://uqer.datayes.com/v3/community/share/5799b908228e5ba291060674
https://kknews.cc/tech/y3a3x8j.html
https://www.itread01.com/content/1543163043.html
