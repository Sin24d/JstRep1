import pandas as pd
import requests as rq
r = rq.get('https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000')
df = pd.read_json(r.text)
df.columns=['Open time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','Ignore']
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

rsarr=[]
rsiarr=[9,11,15]
for n in rsiarr:    
    upmove = []
    downmove = []
    z = 0
    while z < 998:
        if fifth[z + 1] > fifth[z]:
            upmove.append(fifth[z + 1] - fifth[z])
            downmove.append(0)
        else:
            downmove.append(fifth[z] - fifth[z + 1])
            upmove.append(0)
        z += 1

    sumR = []
    sumP = []
    r = 0
    while r < 999 - n:
        sumup = 0
        sumdown = 0
        t = 0
        while t < n:
            sumup += upmove[t + r]
            sumdown += downmove[t + r]
            t += 1
        sumR.append(sumup / n)
        sumP.append(sumdown / n)
        r += 1

    b = 0
    RSIarray = []
    while b < len(sumR):
        RSI = 100 - (100 / (sumR[b] / (sumP[b] - 1)))
        b = b + 1
        RSIarray.append(RSI)
    rsarr.append(RSIarray)
rsarr
#rsarr 9/11/15
print(len(rsarr[0]))

smaarr=[]
SMAnumb=[9,11,15]
for n in SMAnumb:
    arravg = []
    qe = 0
    while qe < len(fifth) - n-1:
        sumnumber = 0
        qw = 0
        while qw < n:
            sumnumber += fifth[qw+qe]
            qw += 1
        avgN = (sumnumber / n)
        arravg.append(avgN)
        qe += 1
    smaarr.append(arravg)
smaarr
#SMAarrey 9/11/15

arraycorr=[first,second,fifth,sixth,seventh,eighth,ninth,tenth,eleventh] 
numbers=[10,2,4]
q=0
rt=0
for a in numbers:
    uno=0
    for x in arraycorr:
        uno+=1
        t=0
        while t<a:
            x.pop(t+rt)
            t+=1
        if scipy.stats.spearmanr(x, smaarr[q])[0]>0.75:
            print('sma',uno, a, scipy.stats.spearmanr(x, smaarr[q])[0])
        if scipy.stats.spearmanr(x, rsarr[q])[0]>0.6:
            print('rsi',uno, a, scipy.stats.spearmanr(x, rsarr[q])[0])
    q+=1
    rt+=a+1
    print(len(x))