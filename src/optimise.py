import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import elo as elo
import glicko as gl
from trueskill import TrueSkill, Rating, quality_1vs1, rate_1vs1
import math
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from run import get_ratings, rolling_validate

elo_params_grid = {"k_factor":hp.quniform("k_factor", 5, 100,5), "rating_class":float,
					"initial":hp.quniform("initial", 800, 1500, 100),
					"beta":hp.quniform("beta", 50, 500, 50),
    				"margin_run":hp.quniform("margin_run",0.01, 0.1, 0.05),
    				"margin_run_norm":hp.quniform("margin_run_norm",10, 100, 10),
    				"margin_wkts":hp.quniform("margin_wkts",0.01, 0.1, 0.05),
    				"k_factor_run":hp.quniform("k_factor_run", 5, 100,5),
    				"k_factor_wkts":hp.quniform("k_factor_wkts", 5, 100,5)}

elo_k_params_grid = {"rating_class":float,
					"initial":hp.quniform("initial", 800, 1500, 100),
					"beta":hp.quniform("beta", 50, 500, 50),
    				"kf_wt_rating":hp.uniform("kf_wt_rating", 0, 0.1),
    				"kf_wt_margin_runs":hp.uniform("kf_wt_margin_runs", 0, 0.1),
    				"kf_wt_margin_wkts":hp.uniform("kf_wt_margin_wkts", 0, 0.1),
    				"kf_wt_winnerby":hp.uniform("kf_wt_winnerby", 0, 0.1),
    				"kf_wt_tossdecision":hp.uniform("kf_wt_tossdecision", 0, 0.1),
    				"kf_wt_tosswinner":hp.uniform("kf_wt_tosswinner", 0, 0.1)}

glicko_params_grid = {"mu":hp.quniform("mu", 800, 1500, 100),
						"phi": hp.quniform("phi", 100, 800, 50),
						"sigma":hp.quniform("sigma", 0.01, 0.09, 0.01),
						"tau":hp.quniform("tau", 0.01, 0.20, 0.01),
						"epsilon":hp.qloguniform("epsilon", 0.0000001, 0.0001, 0.0000001),
						"Q":math.log(10)/400}

trueskill_params_grid = {"mu":hp.quniform("mu", 5, 100, 5),
						"beta": hp.quniform("beta", 1, 10, 0.5),
						"sigma":hp.quniform("sigma", 1, 20, 1),
						"tau":hp.quniform("tau", 0.01, 0.20, 0.01),
						"draw_probability":hp.qloguniform("draw_probability", 0.1, 0.4, 0.1),}

optimise_algo = 'elo_custom_k'
max_evals = 100
def get_err(params):
	ratings = get_ratings(optimise_algo, params)
	err = rolling_validate(ratings, starti=0.50, endi=0.75)
	print "Accuracy: ", 1-err, "  with params: ", params
	return {'loss': err, 'status': STATUS_OK}

trials = Trials()
print('Tuning Parameters')
if optimise_algo=='glicko':
	best = fmin(get_err, glicko_params_grid, algo=tpe.suggest, trials=trials, max_evals=max_evals)
elif optimise_algo=='elo':
	best = fmin(get_err, elo_params_grid, algo=tpe.suggest, trials=trials, max_evals=max_evals)
elif optimise_algo=='trueskill':
	best = fmin(get_err, trueskill_params_grid, algo=tpe.suggest, trials=trials, max_evals=max_evals)
elif optimise_algo=='elo_custom_k':
	best = fmin(get_err, elo_k_params_grid, algo=tpe.suggest, trials=trials, max_evals=max_evals)

print('\n\nBest Scoring Value')
print(best)

final_ratings = get_ratings(optimise_algo, best)
print "Validation Accuracy: ", 1-rolling_validate(final_ratings, starti=0.5, endi=0.75)
print "Test Accuracy: ", 1-rolling_validate(final_ratings, starti=0.75, endi=1)

# Best Scoring Value
# {'margin_run': 0.0, 'k_factor_wkts': 100.0, 'initial': 1100.0, 'margin_run_norm': 80.0, 
# 'beta': 50.0, 'k_factor': 75.0, 'k_factor_run': 95.0, 'margin_wkts': 0.0}
# Validation Accuracy:  0.978328173375
# Test Accuracy:  0.965944272446

# Best Scoring Value
# {'kf_wt_margin_runs': 0.08192817434904767, 'kf_wt_tosswinner': 0.019064236867300233, 'initial': 1200.0, 'kf_wt_rating': 0.07791829445880818, 'beta': 50.0, 'kf_wt_margin_wkts': 0.004559414606145076, 'kf_wt_tossdecision': 0.047789312566250075, 'kf_wt_winnerby': 0.05773614186948804}
# Validation Accuracy:  0.97213622291
# Test Accuracy:  0.965944272446