"""Эталонный порт ядра GAINZ на Python — тем же кодом считаются числа из GAINZ.md.

Зачем: Pine Script нельзя прогнать офлайн, а верить таблице калибровки, не проверив
её ни на чём, кроме самой себя, — плохая идея. Здесь то же ядро (компоненты, скор,
порог, стоп и цели, схема выходов) на синтетических данных, плюс два контроля:
чистое случайное блуждание и нулевая модель со случайными входами.

    python3 tools/gainz_calib.py

Расхождения с pine/GAINZ.pine, о которых стоит знать:
  * старшие таймфреймы и объёмный фильтр здесь считаются на одном ТФ — MTF не эмулируется;
  * комиссия по умолчанию нулевая (costTick=0), чтобы контроли сравнивались честно;
  * пауза после сделки отсчитывается от выхода, как и в индикаторе.
"""
import math, random

def rma(src, n):
    out=[None]*len(src); a=1.0/n; prev=None
    for i,v in enumerate(src):
        if v is None: out[i]=None; continue
        if prev is None:
            if i+1>=n:
                w=[x for x in src[i-n+1:i+1] if x is not None]
                if len(w)==n: prev=sum(w)/n; out[i]=prev
            continue
        prev=a*v+(1-a)*prev; out[i]=prev
    return out

def ema(src,n):
    out=[None]*len(src); a=2.0/(n+1); prev=None
    for i,v in enumerate(src):
        prev=v if prev is None else a*v+(1-a)*prev
        out[i]=prev
    return out

def sma(src,n):
    out=[None]*len(src); s=0.0
    for i,v in enumerate(src):
        s+=v
        if i>=n: s-=src[i-n]
        out[i]=s/n if i>=n-1 else None
    return out

def stdev(src,n):
    out=[None]*len(src)
    for i in range(len(src)):
        if i>=n-1:
            w=src[i-n+1:i+1]; m=sum(w)/n
            out[i]=math.sqrt(sum((x-m)**2 for x in w)/n)
    return out

def hh(src,n):
    return [max(src[max(0,i-n+1):i+1]) for i in range(len(src))]
def ll(src,n):
    return [min(src[max(0,i-n+1):i+1]) for i in range(len(src))]

def clamp(x,a,b): return max(a,min(b,x))
def c01(x): return clamp(x,0.0,1.0)

def gen(n=30000, seed=7):
    """Режимы: тренд / возврат к средней, с кластеризацией волатильности."""
    rnd=random.Random(seed)
    o=h=l=c=100.0; px=100.0; vol=0.6; mode='mr'; left=0; drift=0.0; mean=100.0
    O=[];H=[];L=[];C=[];V=[]
    for i in range(n):
        if left<=0:
            mode = 'trend' if rnd.random()<0.40 else 'mr'
            left = rnd.randint(120,600)
            drift = rnd.choice([-1,1])*rnd.uniform(0.004,0.02)*vol
            mean = px
        left-=1
        vol = max(0.15, vol*0.995 + rnd.gauss(0,0.03) + (0.25 if rnd.random()<0.004 else 0))
        o=px; path=[px]
        for _ in range(6):
            if mode=='trend': step = drift/6 + rnd.gauss(0,vol/2.45)
            else:             step = 0.055*(mean-px)/6 + rnd.gauss(0,vol/2.45)
            px+=step; path.append(px)
        c=px; h=max(path); l=min(path)
        v=abs(c-o)/max(vol,1e-9)*1000+rnd.uniform(200,900)
        O.append(o);H.append(h);L.append(l);C.append(c);V.append(v)
    return O,H,L,C,V

# ── параметры = дефолты индикатора ────────────────────────────────────────────
P=dict(atrLen=14,ctxLen=10,gain=1.25,thrBase=70.0,adaptOn=True,adaptK=0.35,
       wC=.15,wM=.25,wE=.28,wB=.05,wS=.27,engFull=1.60,pinMult=1.50,pinFull=.62,
       minBody=.25,maxBody=2.20,volLen=20,volConf=1.40,momLen=9,momFull=2.20,
       macdF=12,macdS=26,macdSig=9,momTurn=.12,rsiLen=14,rsiBuy=38.,rsiSell=62.,rsiFull=12.,
       stabLen=20,strLen=12,sweepATR=.45,emaFastL=21,emaSlowL=100,tolATR=.60,
       volRegLen=50,volCalm=1.40,volWild=2.80,erLen=20,erLo=.25,erHi=.65,gMin=.35,
       slMult=1.50,swingLen=10,rr1=1.50,rr2=3.00,useTp2=True,maxBars=40,
       cooldown=5,needPat=True,costTick=0.0)

def run(O,H,L,C,V,p=P,thrOverride=None,calThr=None):
    n=len(C)
    tr=[H[0]-L[0]]+[max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1])) for i in range(1,n)]
    atr=rma(tr,p['atrLen'])
    rng=[max(H[i]-L[i],1e-9) for i in range(n)]
    up=[C[i]-C[i-1] if i and C[i]>C[i-1] else 0.0 for i in range(n)]
    dn=[C[i-1]-C[i] if i and C[i]<C[i-1] else 0.0 for i in range(n)]
    au,ad=rma(up,p['rsiLen']),rma(dn,p['rsiLen'])
    rsi=[None if au[i] is None or ad[i] is None else (100.0 if ad[i]==0 else 100-100/(1+au[i]/max(ad[i],1e-12))) for i in range(n)]
    mf,ms=ema(C,p['macdF']),ema(C,p['macdS'])
    macd=[mf[i]-ms[i] for i in range(n)]
    hist=[macd[i]-s for i,s in enumerate(ema(macd,p['macdSig']))]
    ef,es=ema(C,p['emaFastL']),ema(C,p['emaSlowL'])
    volMa=sma(V,p['volLen'])
    rngMa,rngSd=sma(rng,p['stabLen']),stdev(rng,p['stabLen'])
    bodyRat=[abs(C[i]-O[i])/rng[i] for i in range(n)]
    bodyAvg=sma(bodyRat,p['stabLen'])
    lowN,highN=ll(L,p['strLen']),hh(H,p['strLen'])
    swLo,swHi=ll(L,p['swingLen']),hh(H,p['swingLen'])

    er=[0.0]*n
    for i in range(n):
        if i>=p['erLen']:
            den=sum(abs(C[j]-C[j-1]) for j in range(i-p['erLen']+1,i+1))
            er[i]=abs(C[i]-C[i-p['erLen']])/max(den,1e-9)
    warm=max(p['emaSlowL'],p['volRegLen'],p['stabLen'])+p['ctxLen']+5
    atrSma=sma([a if a else 0.0 for a in atr],p['volRegLen'])

    histM,histE,histS=[0.0]*n,[0.0]*n,[0.0]*n   # контекст long
    histM2,histE2,histS2=[0.0]*n,[0.0]*n,[0.0]*n
    comps=[None]*n
    for i in range(n):
        if i<warm or atr[i] is None or rsi[i] is None: continue
        a=max(atr[i],1e-9); bodyN=abs(C[i]-O[i])/a
        upW=H[i]-max(C[i],O[i]); dnW=min(C[i],O[i])-L[i]
        stab=c01(0.60*c01(1-rngSd[i]/max(rngMa[i],1e-9))+0.40*c01(bodyAvg[i]/0.55))
        volRat=a/max(atrSma[i],1e-9)
        gVol=clamp(1-(volRat-p['volCalm'])/max(p['volWild']-p['volCalm'],1e-9),p['gMin'],1.0)
        trend=1 if (ef[i]>es[i] and C[i]>es[i]) else (-1 if (ef[i]<es[i] and C[i]<es[i]) else 0)
        out={}
        for d in (1,-1):
            b0=abs(C[i]-O[i]); b1=abs(C[i-1]-O[i-1])
            eng=(C[i]>O[i] and C[i-1]<O[i-1] and C[i]>=O[i-1] and O[i]<=C[i-1]) if d==1 \
                else (C[i]<O[i] and C[i-1]>O[i-1] and C[i]<=O[i-1] and O[i]>=C[i-1])
            engQ=c01((b0/max(b1,1e-9)-1)/max(p['engFull']-1,0.1)) if eng else 0.0
            wick=dnW if d==1 else upW; opp=upW if d==1 else dnW; mid=(H[i]+L[i])/2
            pin=wick>=p['pinMult']*b0 and wick>opp and (C[i]>mid if d==1 else C[i]<mid)
            pinQ=c01((wick/rng[i])/p['pinFull']) if pin else 0.0
            patQ=max(engQ,pinQ)
            cpos=c01((C[i]-L[i] if d==1 else H[i]-C[i])/rng[i])
            vq=c01((V[i]/volMa[i])/p['volConf']) if volMa[i] else 0.5
            sz=c01(bodyN/max(p['minBody'],.01)) if bodyN<p['minBody'] else (
               c01(1-(bodyN-p['maxBody'])/max(p['maxBody'],.01)) if bodyN>p['maxBody'] else 1.0)
            Cc=c01(sz*(0.55*patQ+0.25*cpos+0.20*vq)) if patQ>0 else 0.0
            roc=(C[i]-C[i-p['momLen']])/a
            mStr=c01((-roc if d==1 else roc)/p['momFull'])
            mT=c01(((hist[i]-hist[i-1]) if d==1 else (hist[i-1]-hist[i]))/max(a*p['momTurn'],1e-9))
            M=c01(0.60*mStr+0.40*mT)
            depth=c01((p['rsiBuy']-rsi[i])/p['rsiFull']) if d==1 else c01((rsi[i]-p['rsiSell'])/p['rsiFull'])
            turn=1.0 if ((rsi[i]>rsi[i-1]) if d==1 else (rsi[i]<rsi[i-1])) else 0.0
            E=c01(0.75*depth+0.25*turn)
            ref=lowN[i-1] if d==1 else highN[i-1]
            pierce=max(0.0,ref-L[i]) if d==1 else max(0.0,H[i]-ref)
            sweepQ=c01(pierce/max(p['sweepATR']*a,1e-9))
            reclaim=(C[i]>ref) if d==1 else (C[i]<ref)
            pull=c01(1-(abs(C[i]-ef[i])/a)/max(p['tolATR'],.01))
            S=c01(max(0.45+0.55*sweepQ if (reclaim and pierce>0) else 0.0, 0.70*pull))
            out[d]=(Cc,M,E,S,patQ,stab,gVol,trend,volRat)
        comps[i]=out
        (m1,e1,s1),(m2,e2,s2)=(out[1][1],out[1][2],out[1][3]),(out[-1][1],out[-1][2],out[-1][3])
        histM[i],histE[i],histS[i]=m1,e1,s1
        histM2[i],histE2[i],histS2[i]=m2,e2,s2

    k=p['ctxLen']
    trades=[]; last=-10**9; pos=None
    for i in range(n):
        if comps[i] is None: continue
        if pos:
            d=pos['d']; risk=pos['risk']
            stopHit = L[i]<=pos['stop'] if d==1 else H[i]>=pos['stop']
            tp1Hit  = H[i]>=pos['tp1']  if d==1 else L[i]<=pos['tp1']
            tp2Hit  = H[i]>=pos['tp2']  if d==1 else L[i]<=pos['tp2']
            r=None
            if stopHit: r=(0.5*p['rr1']+0.5*((pos['stop']-pos['ent'])*d/risk)) if pos['half'] else (pos['stop']-pos['ent'])*d/risk
            elif p['useTp2'] and tp1Hit and not pos['half']:
                pos['half']=True; pos['stop']=pos['ent']
                if tp2Hit: r=0.5*p['rr1']+0.5*p['rr2']
            elif p['useTp2'] and pos['half'] and tp2Hit: r=0.5*p['rr1']+0.5*p['rr2']
            elif not p['useTp2'] and tp1Hit: r=p['rr1']
            elif i-pos['bar']>=p['maxBars']:
                mkt=(C[i]-pos['ent'])*d/risk
                r=(0.5*p['rr1']+0.5*mkt) if pos['half'] else mkt
            if r is not None:
                trades.append((pos['score'],r,i-pos['bar'],pos['d'],pos['live'])); last=i; pos=None
        if pos or i-last<=p['cooldown']: continue
        best=None
        for d in (1,-1):
            Cc,M,E,S,patQ,stab,gVol,trend,volRat=comps[i][d]
            HM=(histM if d==1 else histM2); HE=(histE if d==1 else histE2); HS=(histS if d==1 else histS2)
            Mx=max(HM[max(0,i-k+1):i+1]); Ex=max(HE[max(0,i-k+1):i+1]); Sx=max(HS[max(0,i-k+1):i+1])
            against=(d==1 and trend<0) or (d==-1 and trend>0)
            gT=clamp(1-(er[i]-p['erLo'])/max(p['erHi']-p['erLo'],1e-9),p['gMin'],1.0) if against else 1.0
            G=clamp(min(gVol,gT),p['gMin'],1.0)
            sc=100*p['gain']*G*(p['wC']*Cc+p['wM']*Mx+p['wE']*Ex+p['wB']*stab+p['wS']*Sx)
            if best is None or sc>best[0]: best=(sc,d,patQ,volRat)
        sc,d,patQ,volRat=best
        thr = thrOverride if thrOverride is not None else (
              p['thrBase']*clamp(1+p['adaptK']*(volRat-1),0.75,1.40) if p['adaptOn'] else p['thrBase'])
        eff = min(calThr,thr) if calThr is not None else thr
        if sc>=eff and (patQ>0 or not p['needPat']):
            a=max(atr[i],1e-9); risk=max(p['slMult']*a,1e-9)
            pos=dict(d=d,ent=C[i],stop=C[i]-d*risk,tp1=C[i]+d*p['rr1']*risk,tp2=C[i]+d*p['rr2']*risk,
                     risk=risk,bar=i,score=sc,half=False,live=sc>=thr)
    return trades,n


def walk(n=30000, seed=1):
    """Контроль: случайное блуждание без возврата к средней."""
    r=random.Random(1000+seed); px=100.0; O=[];H=[];L=[];C=[];V=[]; vol=.6
    for _ in range(n):
        vol=max(.15,vol*.995+r.gauss(0,.03)); o=px; path=[px]
        for _ in range(6): px+=r.gauss(0,vol/2.45); path.append(px)
        O.append(o);H.append(max(path));L.append(min(path));C.append(px)
        V.append(abs(px-o)/max(vol,1e-9)*1000+r.uniform(200,900))
    return O,H,L,C,V

def null_model(O,H,L,C,p,n_entries=1200,seed=0):
    """Нулевая модель: случайные входы, та же схема выходов."""
    r=random.Random(seed); n=len(C)
    tr=[H[0]-L[0]]+[max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1])) for i in range(1,n)]
    atr=rma(tr,p['atrLen']); ent=set(r.sample(range(200,n-50),n_entries)); out=[]; pos=None
    for i in range(n):
        if pos:
            d=pos['d']; risk=pos['risk']; res=None
            stopHit=L[i]<=pos['stop'] if d==1 else H[i]>=pos['stop']
            tp1Hit =H[i]>=pos['tp1']  if d==1 else L[i]<=pos['tp1']
            tp2Hit =H[i]>=pos['tp2']  if d==1 else L[i]<=pos['tp2']
            if stopHit:
                res=(0.5*p['rr1']+0.5*((pos['stop']-pos['ent'])*d/risk)) if pos['half'] else (pos['stop']-pos['ent'])*d/risk
            elif tp1Hit and not pos['half']:
                pos['half']=True; pos['stop']=pos['ent']
                if tp2Hit: res=0.5*p['rr1']+0.5*p['rr2']
            elif pos['half'] and tp2Hit: res=0.5*p['rr1']+0.5*p['rr2']
            elif i-pos['bar']>=p['maxBars']:
                mkt=(C[i]-pos['ent'])*d/risk
                res=(0.5*p['rr1']+0.5*mkt) if pos['half'] else mkt
            if res is not None: out.append(res); pos=None
        if pos is None and i in ent and atr[i]:
            risk=max(p['slMult']*max(atr[i],1e-9),1e-9); d=r.choice([1,-1])
            pos=dict(d=d,ent=C[i],stop=C[i]-d*risk,tp1=C[i]+d*p['rr1']*risk,
                     tp2=C[i]+d*p['rr2']*risk,risk=risk,bar=i,half=False)
    return out

def summary(rs):
    if not rs: return "нет сделок"
    wr=100*sum(1 for x in rs if x>0)/len(rs); avg=sum(rs)/len(rs)
    pos=sum(x for x in rs if x>0); neg=-sum(x for x in rs if x<=0)
    return f"n={len(rs):5}  винрейт {wr:5.1f}%  средний {avg:+.3f}R  ПФ {pos/neg if neg else 9.99:4.2f}"

def main():
    seeds=range(1,6)
    allT=[]; bars=0
    for s in seeds:
        O,H,L,C,V=gen(seed=s)
        t,n=run(O,H,L,C,V,calThr=50.0); allT+=t; bars+=n
    print(f"Синтетика со сменой режимов: {bars} баров, {len(allT)} сделок в калибровке\n")
    print(f"{'ДИАПАЗОН':>9} {'СДЕЛОК':>7} {'ВИНРЕЙТ':>8} {'ПФ':>5} {'СРЕДНИЙ R':>10} {'БАРОВ':>6}")
    for lo,hi,nm in [(0,50,'< 50'),(50,58,'50-57'),(58,66,'58-65'),(66,74,'66-73'),(74,82,'74-81'),(82,999,'82-100')]:
        b=[x for x in allT if lo<=x[0]<hi]
        if not b: print(f"{nm:>9} {0:>7}"); continue
        wr=100*sum(1 for x in b if x[1]>0)/len(b); avg=sum(x[1] for x in b)/len(b)
        pos=sum(x[1] for x in b if x[1]>0); neg=-sum(x[1] for x in b if x[1]<=0)
        print(f"{nm:>9} {len(b):>7} {wr:>7.1f}% {pos/neg if neg else 9.99:>5.2f} {avg:>10.3f} "
              f"{sum(x[2] for x in b)/len(b):>6.1f}")
    live=[x[1] for x in allT if x[4]]; sub=[x[1] for x in allT if not x[4]]
    print(f"\n  от рабочего порога: {summary(live)}")
    print(f"  ниже порога:        {summary(sub)}")
    print(f"  частота боевых сигналов: 1 на {bars/max(len(live),1):.0f} баров")

    ctrl=[]; null=[]
    for s in seeds:
        O,H,L,C,V=walk(seed=s)
        ctrl+=[x[1] for x in run(O,H,L,C,V)[0]]
        null+=null_model(O,H,L,C,P,seed=s)
    print("\nКонтроль на случайном блуждании (возврата к средней в данных нет):")
    print(f"  сигналы движка:  {summary(ctrl)}")
    print(f"  случайные входы: {summary(null)}")
    print("\nЕсли эти две строки близки — на бесструктурных данных движок не выдумывает эдж,")
    print("а плюс нулевой модели показывает, сколько даёт сама схема выходов.")

if __name__=='__main__':
    main()
