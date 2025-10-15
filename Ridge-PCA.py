import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read a CSV file into a DataFrame
df = pd.read_csv('parallel_1.txt', sep='\t')
pd.set_option('display.max_columns', None)

# Convert to datetime
df['datetime'] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%d %H:%M:%S.%f')

# Subtract the first datetime to get a timedelta
df['elapsed_seconds'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

injection_start_time = 60
injection_rate = 0.005   #mL/sec (300uL/min)

df['fluid_volume'] = np.where(
    df['elapsed_seconds'] < injection_start_time,
    0.0,
    (df['elapsed_seconds'] - injection_start_time)*injection_rate
)

smooth = signal.savgol_filter(df.iloc[:, 60:430], window_length=11, polyorder=2)
print(smooth.shape)

y = np.round(df['fluid_volume'].values[:smooth.shape[0]], 2)

scaler = StandardScaler()
data = scaler.fit_transform(smooth)
pca = PCA(n_components=6)
data = pca.fit_transform(data, y)

#Perpendicular 1
groups = np.concatenate([
    np.full(556, 0),
    np.full(907, 1),
    np.full(909, 2),
    np.full(908, 3),
    np.full(909, 4),
    np.full(909, 5),
    np.full(1040, 6)
])

#Perpendicular 2
groups = np.concatenate([
    np.full(554, 0),
    np.full(683, 1),
    np.full(907, 2),
    np.full(907, 3),
    np.full(908, 4),
    np.full(907, 5),
    np.full(1018, 6)
])

#Parallel
groups = np.concatenate([
    np.full(554, 0),
    np.full(908, 1),
    np.full(907, 2),
    np.full(908, 3),
    np.full(908, 4),
    np.full(907, 5),
    np.full(905, 6)
])

train_group_ids = [0, 1, 2, 3, 5]
test_group_ids = [4, 6]

train_mask = np.isin(groups, train_group_ids)
test_mask = np.isin(groups, test_group_ids)

X_train, X_test = data[train_mask], data[test_mask]
Y_train, Y_test = y[train_mask], y[test_mask]


# Cross-validation setup: Venetian Blind
class VenetianBlindCV(BaseCrossValidator):
    def __init__(self, n_splits=5, thickness=20):
        self.n_splits = n_splits
        self.thickness = thickness

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Assign samples to a fold based on their position in the blinds
        fold_assignments = np.full(n_samples, -1)
        i = 0
        while i < n_samples:
            for fold in range(self.n_splits):
                end = i + self.thickness
                if i >= n_samples:
                    break
                fold_assignments[i:end] = fold
                i = end

        for fold in range(self.n_splits):
            test_mask = fold_assignments == fold
            train_mask = ~test_mask
            yield indices[train_mask], indices[test_mask]


cv = VenetianBlindCV(n_splits=5, thickness=50)



#Ridge
RidgeReg = Ridge()
# Define the parameter grid for optimization
param_grid = {
    'alpha': [1, 10, 100],
    'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
}

# Define scoring metrics
scoring = {
    'r2': 'r2'
}

# Perform Grid Search with Multiple Scoring Metrics
grid_search = GridSearchCV(RidgeReg, param_grid, cv=cv, scoring=scoring, refit='r2', n_jobs=-1, verbose=3)
grid_search.fit(X_train, Y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best r2:", grid_search.best_score_)

# Train the best model
best_ridge = grid_search.best_estimator_
r2_scores = []
mse = []

for train_idx, val_idx in cv.split(X_train, Y_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = Y_train[train_idx], Y_train[val_idx]

    best_ridge.fit(X_tr, y_tr)
    y_pred = best_ridge.predict(X_val)

    r = r2_score(y_val, y_pred)
    m = mean_squared_error(y_val, y_pred)

    r2_scores.append(r)
    mse.append(m)
    #print(X_tr.shape, X_val.shape)

# Print mean and std metrics
print("=== Cross-Validation Performance ===")
#print(r2_scores)
#print(mse)
mean_r2_score = np.mean(r2_scores)
mean_mse = np.mean(mse)
print("%0.4f ± %0.4f" % (np.mean(r2_scores),np.std(r2_scores)))
print("%0.4f ± %0.4f" % (np.mean(mse),np.std(mse)))

# Prediction and evaluation
Y_pred = best_ridge.predict(X_test).flatten()
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
rmse = (mse)**(1/2)
mae =  mean_absolute_error(Y_test, Y_pred)
print("=== Testing Performance ===")
print(f'R² Score: {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

# === Plot: True vs Predicted ===
plt.figure(figsize=(6, 5))
plt.scatter(Y_test, Y_pred, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('True Volume (mL)')
plt.ylabel('Predicted Volume (mL)')
plt.title('True vs Predicted Fluid Volume')
plt.grid(True)
plt.tight_layout()
plt.show()