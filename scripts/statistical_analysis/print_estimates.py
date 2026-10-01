"""
Print the posterior estimates from the fitted models across all image sets.

Behavioral sets (set1-3) use the human Likert 'naturalness'; the prediction
sets use the ViTNat 'vitnat'.

For linear regressions, reports β, 96% HPDI, and P(β < 0 | data) or 
P(β > 0 | data) for the primary predictor beta (beta_0) of each:
- naturalness -> jpeg_based_compressibility
- naturalness -> canny_based_compressibility
- naturalness -> crr (memorability)
- jpeg_based_compressibility -> crr
- canny_based_compressibility -> crr

For the mediation models, reports the direct effect (cprime), indirect effect
(a*b), and total effect (c = cprime + a*b), along with their 96% HPDI and
P(parameter < 0 | data) or P(parameter > 0 | data) for each:
- naturalness -> jpeg_based_compressibility -> crr
- naturalness -> canny_based_compressibility -> crr

Also reports the same linear-regression battery for the two sets with a
group-level random intercept (linear_regressions_random_intercept.py): set3
(grouped by category) and memcat (grouped by subcategory).
"""

import sys
sys.path.append('../../src')

import arviz as az
from arviz_stats import hdi

from paths import IDATA_PATH
from misc import naturalness_column
from linear_regressions import FILENAME_ALIAS

BEHAVIORAL_SETS = ['set1', 'set2', 'set3']
PREDICTION_SETS = ['isola', 'memcat', 'lamem']
RANDOM_INTERCEPT_SETS = ['set3', 'memcat']

# (predictor, outcome, reported tail); 'naturalness' is resolved per set.
# Compressibility effects are positive, memorability effects negative.
# The reported tail is the one opposite the hypothesis, i.e. how much posterior
# mass contradicts it (the Bayesian analogue of a one-tailed p-value): a
# hypothesized-positive effect reports P(β < 0 | data), and vice versa.
REGRESSIONS = [
    ('naturalness', 'jpeg_based_compressibility', '<'),
    ('naturalness', 'canny_based_compressibility', '<'),
    ('naturalness', 'crr', '>'),
    ('jpeg_based_compressibility', 'crr', '>'),
    ('canny_based_compressibility', 'crr', '>'),
]

# (mediator) for naturalness -> mediator -> crr. The direct effect (cprime), the
# indirect effect (a*b) and their sum, the total effect (c), are all hypothesized
# to be negative (natural scenes are less memorable, directly and via greater
# compressibility), so all report the opposite tail, P(parameter > 0).
MEDIATIONS = [
    ('jpeg_based_compressibility', '>'),
    ('canny_based_compressibility', '>'),
]

def print_effect(image_set, path, var_name, direction, symbol='β'):
    """Print a manuscript-ready line for the posterior of `var_name` in `path`.
    symbol names the estimate in the output (e.g. β, c', a*b); direction selects
    the reported tail: '<' for P(symbol < 0), '>' for P(symbol > 0)."""
    if not path.exists():
        print(f'  {image_set:<8s} [missing: {path.name}]')
        return
    samples = az.from_netcdf(path)['posterior'].to_dataset()[var_name].values
    lo, hi = hdi(samples.ravel(), prob=0.96)
    p = (samples > 0).mean() if direction == '>' else (samples < 0).mean()
    label = f'P({symbol} {direction} 0 | data)'
    if p < 1e-4:
        p_str = f'{label} < .0001'
    elif p > 1 - 1e-4:
        p_str = f'{label} > .9999'
    else:
        p_str = f'{label} = ' + f'{p:.4f}'.lstrip('0')
    print(f'  {image_set:<8s} {symbol} = {samples.mean():.2f}, '
          f'96% HPDI = [{lo:.2f}, {hi:.2f}], {p_str}')

def print_battery(image_sets):
    for predictor, outcome, direction in REGRESSIONS:
        # A battery is one homogeneous set group, so the header label matches every row.
        nat0 = naturalness_column(image_sets[0])
        print(f'{nat0 if predictor == "naturalness" else predictor} -> {outcome}')
        for image_set in image_sets:
            resolved_predictor = naturalness_column(image_set) if predictor == 'naturalness' else predictor
            path = IDATA_PATH / f'{image_set}_lr_o_{FILENAME_ALIAS[outcome]}_p_{FILENAME_ALIAS[resolved_predictor]}.nc'
            print_effect(image_set, path, 'beta_0', direction)
        print()

def print_mediation_battery(image_sets):
    for mediator, direction in MEDIATIONS:
        nat0 = naturalness_column(image_sets[0])
        print(f'{nat0} -> {mediator} -> crr (mediation)')
        for effect_var, label, symbol in [('cprime', 'direct effect', "c'"),
                                          ('indirect_effect', 'indirect effect', 'a*b'),
                                          ('total_effect', 'total effect', 'c')]:
            print(f'  {label} ({symbol}):')
            for image_set in image_sets:
                p, m, o = FILENAME_ALIAS[naturalness_column(image_set)], FILENAME_ALIAS[mediator], FILENAME_ALIAS['crr']
                path = IDATA_PATH / f'{image_set}_med_p_{p}_m_{m}_o_{o}.nc'
                print_effect(image_set, path, effect_var, direction, symbol=symbol)
        print()

def print_random_intercept_battery():
    for predictor, outcome, direction in REGRESSIONS:
        print(f'{predictor} -> {outcome} (random intercept)')
        for image_set in RANDOM_INTERCEPT_SETS:
            resolved_predictor = naturalness_column(image_set) if predictor == 'naturalness' else predictor
            path = IDATA_PATH / f'{image_set}_lrri_o_{FILENAME_ALIAS[outcome]}_p_{FILENAME_ALIAS[resolved_predictor]}.nc'
            print_effect(image_set, path, 'beta_0', direction)
        print()

def main():
    print_battery(BEHAVIORAL_SETS)
    print_mediation_battery(BEHAVIORAL_SETS)
    print('=' * 60 + '\n')
    print_battery(PREDICTION_SETS)
    print_mediation_battery(PREDICTION_SETS)
    print('=' * 60 + '\n')
    print_random_intercept_battery()

if __name__ == '__main__':
    main()
