from bdse import get_conftools_bdse_estimators, get_conftools_bdse_models
from comptests import comptests_for_all, comptests_for_all_pairs

__all__ = [
   'for_all_bdse_estimators',
   'for_all_bdse_models',    
   'for_all_bdse_models_estimators',
]

models = get_conftools_bdse_models()
estimators = get_conftools_bdse_estimators()

for_all_bdse_models = comptests_for_all(models)
for_all_bdse_estimators = comptests_for_all(estimators)
for_all_bdse_models_estimators = comptests_for_all_pairs(models, estimators)


for_all_bdse_examples = for_all_bdse_models