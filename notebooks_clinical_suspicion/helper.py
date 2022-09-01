import numpy as np

def pecarn_rule_predict(d, o='iai_intervention', verbose=False):
    '''Predict each subgroup in the original PECARN rule
    Return the predictions of the PECARN rule
    '''
    n = d.shape[0]
    npos = d[o].sum()
    if verbose:
        print(f'{"Initial":<25} {npos} / {n}')
    rules = [
        ('AbdTrauma_or_SeatBeltSign', ['yes']),
        ('GCSScore', range(14)),
        ('AbdTenderDegree', ['Mild', 'Moderate', 'Severe']),
        ('ThoracicTrauma', ['yes']),        
        ('AbdomenPain', ['yes']),
        ('DecrBreathSound', ['yes']),
        ('VomitWretch', ['yes']),
    ]
    preds_high_risk = np.zeros(n).astype(bool)
    d_small = d
    for rule in rules:
        k, vals = rule
        preds_high_risk_new = d[k].isin(vals)
        preds_high_risk = preds_high_risk | preds_high_risk_new 

        if verbose:
            idxs_high_risk = d_small[k].isin(vals)
            do = d_small[idxs_high_risk]
            d_small = d_small[~idxs_high_risk]
            num2 = do[o].sum()
            denom2 = do.shape[0]
            print(f'{k:<25} {d_small[o].sum():>3} / {d_small.shape[0]:>5}\t{num2:>3} / {denom2:>4} ({num2/denom2*100:0.1f})')
    return preds_high_risk
    