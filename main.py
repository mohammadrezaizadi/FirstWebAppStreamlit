import streamlit as st
from PIL import Image
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, date
import tradeHelper as th
import json
from streamlit_lightweight_charts import renderLightweightCharts
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pandas_ta as ta


def signal(data, gapp, position):
    data = data.assign(sb=None)
    data = data.assign(ss=None)
    # dataFrame['mv34'] = mv34
    # dataFrame['mv55'] = mv55
    # dataFrame['mv62'] = mv62
    signalBuy = []
    signalSell = []
    f = -1

    counter = 0

    # print(dataFrame.iloc[100])
    # print(data.values)
    # print(data[:1].dtypes)

    for i in range(len(data)):
        if (data['mv34'].iloc[i] > data['mv55'].iloc[i] and
                data['mv34'].iloc[i] > data['mv62'].iloc[i] and
                data['MACD_6_12_5'].iloc[i] > 0):
            if f != 1:
                data['sb'].iloc[i + position] = (data['close'].iloc[i + position]) + gapp
                f = 1

        elif (data['mv34'].iloc[i] < data['mv55'].iloc[i] and
              data['mv34'].iloc[i] < data['mv62'].iloc[i] and
              data['MACD_6_12_5'].iloc[i] < 0):
            if f != 0:
                data['ss'].iloc[i + position] = (data['close'].iloc[i + position]) - gapp
                f = 0

    return data


def get_symbols(name):
    return th.Getsymbols(name)


def get_data(symbol_Name, timeframe, days):
    symb = th.Getsymbols(symbol_Name)[0].name
    today = datetime.combine(date.today(), datetime.min.time())
    start = today + timedelta(days=-days)
    c = th.GetCandels(symb, start, today, timeframe)
    df = pd.DataFrame(c, index=pd.DatetimeIndex(pd.to_datetime(c['time'], unit='s')))
    # dataFrame.set_index(pd.DatetimeIndex(pd.to_datetime(c['time'], unit='s')))
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


def get_movings(df, args=[]):
    for x in args:
        mv = int(x)
        df['mv' + str(x)] = df['close'].rolling(window=mv).mean()
    return df


def get_MFI(df, per):
    typical_price = (df['close'] + df['high'] + df['low']) / 3
    money_flow = typical_price * df['tick_volume']
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.append(money_flow.iloc[i - 1])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
            positive_flow.append(0)
            negative_flow.append(money_flow.iloc[i - 1])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = []
    negative_mf = []

    for i in range(per - 1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i + 1 - per:i + 1]))

    for i in range(per - 1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i + 1 - per:i + 1]))

    mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf)))

    for i in range(0, per):
        mfi = np.append(mfi, [.0])
    df['MFI'] = mfi
    return df


# ===================================== properies
timeframe = 'h1'
days_GetData = 100
gapp = .005
position = 0
symbolName = 'EURUSD_i'
period = 14
COLOR_BULL = 'rgba(38,166,154,0.9)'  # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
days_prediction = 5

# ========================================= site began


st.sidebar.header('تنظیمات ')
symbolName = st.sidebar.selectbox('نام نماد', pd.DataFrame(get_symbols(''))[93])
days_GetData = st.sidebar.text_input('تعداد روز ', value=100)
vals = ['m1', 'm5', 'm15', 'm30', 'h1', 'h4', 'd1']
timeframe = st.sidebar.selectbox('تایم فریم ', vals, index=vals.index('h1'))
days_prediction = st.sidebar.number_input('تعداد کندل های پیش بینی ', value=5)
mydf = get_data(get_symbols(symbolName)[0].name, timeframe, int(days_GetData))
mydf = get_movings(mydf, [34, 55, 62, 200])
mydf = get_MFI(mydf, period)
mydf.ta.macd(close='close', fast=6, slow=12, signal=5, append=True)  # calculate macd
mydf.ta.adx(close='close', offset=14, append=True)  # calculate adx
mydf = signal(mydf, gapp, position)

st.write('''
# تحلیل داده های بورسی
this is **stock** visualizer

''')

img = Image.open('logo.jpg')
st.image(img, width=350, caption='تحلیل داده های بورسی ')

# symbol = 'GOOG'
# data = yf.Ticker(symbol)
# history = data.history(period='1d', start='2020-02-01', end="2020-9-10")
# ====================== line chart

h = ' قیمت' + str(days_GetData) + 'روز گذشته ' + 'تایم فریم ' + timeframe
st.markdown('<h3 style="direction:rtl">' + h + '</h3>', unsafe_allow_html=True)
st.line_chart(mydf['close'])
st.header('حجم معاملات ')
st.line_chart(mydf['tick_volume'])
st.write(mydf.describe())
# mydf = mydf.reset_index()
# ======================chart

candles = json.loads(mydf.to_json(orient="records"))
volume = json.loads(mydf.rename(columns={"tick_volume": "value", }).to_json(orient="records"))
mydf['color'] = np.where(mydf['open'] > mydf['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear
macd_fast = json.loads(mydf.rename(columns={"MACDh_6_12_5": "value"}).to_json(orient="records"))
macd_slow = json.loads(mydf.rename(columns={"MACDs_6_12_5": "value"}).to_json(orient="records"))
# mydf['color'] = np.where(mydf['MACD_6_12_5'] > 0, COLOR_BULL, COLOR_BEAR)  # MACD histogram color
macd_hist = json.loads(mydf.rename(columns={"MACD_6_12_5": "value"}).to_json(orient="records"))
adx = json.loads(mydf.rename(columns={"ADX_14": "value", 'color': 'col32'}).to_json(orient="records"))
adx_dmp = json.loads(mydf.rename(columns={"DMP_14": "value", 'color': 'col32'}).to_json(orient="records"))
adx_dmn = json.loads(mydf.rename(columns={"DMN_14": "value", 'color': 'col32'}).to_json(orient="records"))
mv34 = json.loads(mydf.rename(columns={"mv34": "value", 'color': 'col32'}).to_json(orient="records"))
mv55 = json.loads(mydf.rename(columns={"mv55": "value", 'color': 'col32'}).to_json(orient="records"))
mv62 = json.loads(mydf.rename(columns={"mv62": "value", 'color': 'col32'}).to_json(orient="records"))

# =======================markers
df2 = mydf.loc[mydf['sb'].values != None, ['time']]
df2 = df2.assign(position='belowBar')
df2 = df2.assign(color='green')
df2 = df2.assign(shape='arrowUp')
df2 = df2.assign(text='buy')
df2 = df2.assign(size=3)
df3 = mydf.loc[mydf['ss'].values != None, ['time']]
df3 = df3.assign(position='aboveBar')
df3 = df3.assign(color='red')
df3 = df3.assign(shape='arrowDown')
df3 = df3.assign(text='sell')
df3 = df3.assign(size=3)
df2 = pd.concat([df2, df3], axis=0, join='outer')
marker = json.loads(df2.to_json(orient="records"))
# =======================markers

# =================================== prediction
forecast = int(days_prediction)
mydf = mydf.sort_values('time', ascending=False)

temp = mydf[['close']].head(n=forecast)
df = mydf[['close']].tail(n=-forecast)
# print('df len ---->>' , len(df))
print('temp = ----->' , temp)
st.line_chart(temp)

df['Prediction'] = df[['close']].shift(-forecast)
print('first df : ---------------> ' , df)

x = np.array(df.drop(columns=['close']))
x = x[:-forecast]
# print(x)
y = np.array(df['Prediction'])
y = y[:-forecast]
# print(y)


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
mysvr = SVR(kernel='rbf', C=1000, gamma=0.1)
mysvr.fit(xtrain, ytrain)
svmconf = mysvr.score(xtest, ytest)
st.header('SVM Accuracy')
st.success(svmconf)



x_forecast=np.array(df.drop(columns=['Prediction']))[-forecast:]
svmpred=mysvr.predict(x_forecast)
st.header('SVM Prediction')
st.success(svmpred)
print(svmpred)
st.line_chart(svmpred)

lr=LinearRegression()
lr.fit(xtrain,ytrain)
lrconf=lr.score(xtest,ytest)
st.header('LR Accuracy')
st.success(lrconf)

lrpred=lr.predict(x_forecast)
st.header('LR Prediction')
st.success(lrpred)
print(lrpred)

st.line_chart(lrpred)

# =================================== prediction


# ========================= shw chart

chartMultipaneOptions = [
    {
        "width": 650,
        "height": 400,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "grid": {
            "vertLines": {
                "color": "rgba(197, 203, 206, 0.5)"
            },
            "horzLines": {
                "color": "rgba(197, 203, 206, 0.5)"
            }
        },
        "crosshair": {
            "mode": 0
        },
        "priceScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)"
        },
        "timeScale": {
            "borderColor": "rgba(197, 203, 206, 0.8)",
            "barSpacing": 15
        },
        "watermark": {
            "visible": True,
            "fontSize": 48,
            "horzAlign": 'center',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.3)',
            "text": symbolName + ' ' + timeframe,
        }
    },
    {
        "width": 650,
        "height": 100,
        "layout": {
            "background": {
                "type": 'solid',
                "color": 'transparent'
            },
            "textColor": 'black',
        },
        "grid": {
            "vertLines": {
                "color": 'rgba(42, 46, 57, 0)',
            },
            "horzLines": {
                "color": 'rgba(42, 46, 57, 0.6)',
            }
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'top',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'Volume',
        }
    },
    {
        "width": 650,
        "height": 200,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'MACD',
        }
    },
    {
        "width": 650,
        "height": 200,
        "layout": {
            "background": {
                "type": "solid",
                "color": 'white'
            },
            "textColor": "black"
        },
        "timeScale": {
            "visible": False,
        },
        "watermark": {
            "visible": True,
            "fontSize": 18,
            "horzAlign": 'left',
            "vertAlign": 'center',
            "color": 'rgba(171, 71, 188, 0.7)',
            "text": 'ADX',
        }
    }
]

seriesCandlestickChart = [
    {
        "type": 'Candlestick',
        "data": candles,
        "options": {
            "upColor": COLOR_BULL,
            "downColor": COLOR_BEAR,
            "borderVisible": False,
            "wickUpColor": COLOR_BULL,
            "wickDownColor": COLOR_BEAR
        },
        "markers": marker
    },
    {
        "type": 'Line',
        "data": mv34,
        "options": {
            "color": "green",
            "lineWidth": 1,
            "upColor": "green",
            "downColor": "green",
            "wickUpColor": "green",
            "wickDownColor": "green"
        }
    },
    {
        "type": 'Line',
        "data": mv55,
        "options": {
            "color": "yellow",
            "lineWidth": 1,
            "upColor": "yellow",
            "downColor": "yellow",
            "wickUpColor": "yellow",
            "wickDownColor": "yellow"
        }
    },
    {
        "type": 'Line',
        "data": mv62,
        "options": {
            "color": "red",
            "lineWidth": 1,
            "upColor": "red",
            "downColor": "red",
            "wickUpColor": "red",
            "wickDownColor": "red"
        }
    }
]

seriesVolumeChart = [
    {
        "type": 'Histogram',
        "data": volume,
        "options": {
            "priceFormat": {
                "type": 'volume',
            },
            "priceScaleId": ""  # set as an overlay setting,
        },
        "priceScale": {
            "scaleMargins": {
                "top": 0,
                "bottom": 0,
            },
            "alignLabels": False
        }
    }
]
seriesMACDchart = [
    {
        "type": 'Line',
        "data": macd_fast,
        "options": {
            "color": 'blue',
            "lineWidth": 2
        }
    },
    {
        "type": 'Line',
        "data": macd_slow,
        "options": {
            "color": 'green',
            "lineWidth": 2
        }
    },
    {
        "type": 'Histogram',
        "data": macd_hist,
        "options": {
            "color": 'red',
            "lineWidth": 1
        }
    }
]
seriesADXchart = [
    {
        "type": 'Line',
        "data": adx,
        "options": {
            "color": 'black',
            "lineWidth": 2
        }
    },
    {
        "type": 'Line',
        "data": adx_dmn,
        "options": {
            "color": 'red',
            "lineWidth": 1
        }
    },
    {
        "type": 'Line',
        "data": adx_dmp,
        "options": {
            "color": 'green',
            "lineWidth": 1
        }
    }
]
st.subheader("نمودار ها")

renderLightweightCharts([
    {
        "chart": chartMultipaneOptions[0],
        "series": seriesCandlestickChart
    },
    {
        "chart": chartMultipaneOptions[1],
        "series": seriesVolumeChart
    },

    {
        "chart": chartMultipaneOptions[2],
        "series": seriesMACDchart
    },

    {
        "chart": chartMultipaneOptions[3],
        "series": seriesADXchart
    },
], 'multipane')
