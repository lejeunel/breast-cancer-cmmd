#!/usr/bin/env ipython

from sklearn.metrics import roc_auc_score
import numpy as np

size = 100
y_true = np.random.binomial(1, p=0.68, size=size)
y = np.random.uniform(0.5, 1, size=size)
# y = np.zeros(size, dtype=int)

print(f"y: {y}")
print(f"y_true: {y_true}")

auc_roc = roc_auc_score(y_true, y)

print(f"auc-roc: {auc_roc}")
