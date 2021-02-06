import pandas as pd
import requests as rq
r = rq.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000')
df = pd.read_json(r.text)
df.columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore']
#df.drop(columns=['Ignore'])
df.describe()

first=df.loc[0:len(df),'Open time']
second=df.loc[0:len(df),'Open']
third=df.loc[0:len(df),'High']
fourth=df.loc[0:len(df),'Low']
fifth=df.loc[0:len(df),'Close']
sixth=df.loc[0:len(df),'Volume']
seventh=df.loc[0:len(df),'Close time']
eighth=df.loc[0:len(df),'Quote asset volume']
ninth=df.loc[0:len(df),'Number of trades']
tenth=df.loc[0:len(df),'Taker buy base asset volume']
eleventh=df.loc[0:len(df),'Taker buy quote asset volume']

import scipy.stats
arraycorr=[first,second,fifth,sixth,seventh,eighth,ninth,tenth,eleventh]
#print(scipy.stats.pearsonr(first, second)[0])
#print(scipy.stats.pearsonr(second, sixth))
#print(scipy.stats.pearsonr(sixth, fourth))
o=0
for i in arraycorr:
    o+=1
    oi=0
    for u in arraycorr:
        oi+=1
        if scipy.stats.pearsonr(i, u)[0]>0.7:
            print(o, oi, scipy.stats.pearsonr(i, u))

q=0
for w in arraycorr:
    q+=1
    io=0
    for e in arraycorr:
        io+=1
        if scipy.stats.spearmanr(w, e)[0]>0.7:
            print(q, io, scipy.stats.spearmanr(w, e)[0])

q=0
for w in arraycorr:
    q+=1
    io=0
    for e in arraycorr:
        io+=1
        #if scipy.stats.linregress(w, e)[0]>0.7:
        print(q, io, scipy.stats.linregress(w, e))

