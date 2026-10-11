#!/usr/bin/env python3
"""Read-only out-of-time holdout sensitivity for null market_index batch imputation.

Compare the same Jan–Aug trained model on Sep–Oct loads. Both policies use only
unlabeled Sep–Oct FEATURES at inference (same-date policy); neither uses their
posted_rate for features or model fitting. Targets are accessed only for metrics.
The ~1.5% flagged rows are excluded only from CLEAN reporting, never from ALL.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from freight_rate_model import fit, predict


def metrics(y,p):
    return dict(n=len(y), MAE=np.abs(y-p).mean(), RMSE=np.sqrt(np.mean((y-p)**2)),
                MAPE=100*np.mean(np.abs(y-p)/y),
                R2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',default='data/train-test.csv')
    parser.add_argument('--out',default='reports/validation_signal_sensitivity.csv')
    args=parser.parse_args()
    df=pd.read_csv(args.train,parse_dates=['date'])
    tr,te=df[df.date<'2025-09-01'],df[df.date>='2025-09-01']
    model=fit(tr)
    y=te.posted_rate.to_numpy(float)
    p_train=predict(model,te)
    p_daily=predict(model,te,same_period=te)
    # The SAME test-label clean mask is used for both policies; never for fitting.
    clean=np.abs(np.log(y)-np.log(p_train))<.4
    clean_daily=np.abs(np.log(y)-np.log(p_daily))<.4
    print('clean masks identical:',np.array_equal(clean,clean_daily))
    assert np.array_equal(clean,clean_daily),'inspect discrepancy before comparing'
    miss=te.market_index.isna().to_numpy()
    rows=[]
    for name,pred in [('training-only median',p_train),('same-date test-feature median',p_daily)]:
        for grp,mask in [('clean',clean),('ALL',np.ones(len(y),dtype=bool))]:
            rows.append({'policy':name,'evaluation':grp,'null_market_index':int(miss.sum()),
                         **metrics(y[mask],pred[mask])})
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    tab=pd.DataFrame(rows);tab.to_csv(out,index=False)
    print('Sep–Oct labeled holdout, feature-availability sensitivity:')
    print(tab.round(5).to_string(index=False))
    delta=np.abs(p_daily-p_train)
    print('affected prediction rows:',int((delta>0.005).sum()),
          '| max absolute prediction delta:',round(delta.max(),2))
    print('wrote:',out,'| no submission CSV modified')


if __name__=='__main__':main()
