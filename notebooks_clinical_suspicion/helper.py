import numpy as np

def pecarn_rule_predict(d, o='iai_intervention', verbose=False):
    '''Predict each subgroup in the original PECARN rule
    Return the probabilistic predictions of the PECARN rule and the threshold binary predictions
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
    preds_probabilistic = np.zeros(n)
    d_small = d
    for rule in rules:
        k, vals = rule
        preds_high_risk_new = d[k].isin(vals)
        idxs_sub_category = preds_high_risk_new & (~preds_high_risk) # this is the new cohort of patients that is high risk
        preds_probabilistic[idxs_sub_category] = np.mean(d[idxs_sub_category][o])
        preds_high_risk = preds_high_risk | preds_high_risk_new 

        if verbose:
            idxs_high_risk = d_small[k].isin(vals)
            do = d_small[idxs_high_risk]
            d_small = d_small[~idxs_high_risk]
            num2 = do[o].sum()
            denom2 = do.shape[0]
            print(f'{k:<25} {d_small[o].sum():>3} / {d_small.shape[0]:>5}\t{num2:>3} / {denom2:>4} ({num2/denom2*100:0.1f})')
    preds_probabilistic[~preds_high_risk] = np.mean(d[~preds_high_risk][o])        
    return preds_probabilistic, preds_high_risk.astype(int)
    