import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.preprocessing import StandardScaler

# Read a CSV file into a DataFrame
df_train = pd.read_csv('Perpendicular_1.txt', sep='\t')
df_test = pd.read_csv('Perpendicular_2.txt', sep='\t')


df_train['datetime'] = pd.to_datetime(df_train.iloc[:, 0], format='%Y-%m-%d %H:%M:%S.%f')
df_train['elapsed_seconds'] = (df_train['datetime'] - df_train['datetime'].iloc[0]).dt.total_seconds()

df_test['datetime'] = pd.to_datetime(df_test.iloc[:, 0], format='%Y-%m-%d %H:%M:%S.%f')
df_test['elapsed_seconds'] = (df_test['datetime'] - df_test['datetime'].iloc[0]).dt.total_seconds()

injection_start_time = 60
injection_rate = 0.005   #mL/sec (300uL/min)

df_train['fluid_volume'] = np.where(
    df_train['elapsed_seconds'] < injection_start_time,
    0.0,
    (df_train['elapsed_seconds'] - injection_start_time)*injection_rate
)
print(df_train[['elapsed_seconds', 'fluid_volume']])

df_test['fluid_volume'] = np.where(
    df_test['elapsed_seconds'] < injection_start_time,
    0.0,
    (df_test['elapsed_seconds'] - injection_start_time)*injection_rate
)
print(df_test[['elapsed_seconds', 'fluid_volume']])

X_train = signal.savgol_filter(df_train.iloc[:, 60:430], window_length=11, polyorder=2)
print(X_train.shape)

X_test = signal.savgol_filter(df_test.iloc[:, 60:430], window_length=11, polyorder=2)
print(X_test.shape)

Y_train = np.round(df_train['fluid_volume'].values[:X_train.shape[0]], 2)
print(Y_train.shape)
Y_test = np.round(df_test['fluid_volume'].values[:X_test.shape[0]], 2)
print(Y_test.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


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

#SVR
svr = SVR()

param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'C': [0.1, 1]
}

scoring = {
    'r2': 'r2'
}

# Perform Grid Search with Multiple Scoring Metrics
grid_search = GridSearchCV(svr, param_grid, cv=cv, scoring=scoring, refit='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, Y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)
print("Best r2:", grid_search.best_score_)

# Train the best model
best_svr = grid_search.best_estimator_

r2_scores = []
mse = []

best_svr = svr.fit(X_train, Y_train)
print('before CV')
for train_idx, val_idx in cv.split(X_train, Y_train):
    print('starting cv')
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = Y_train[train_idx], Y_train[val_idx]

    best_svr.fit(X_tr, y_tr)
    y_pred = best_svr.predict(X_val)

    r = r2_score(y_val, y_pred)
    m = mean_squared_error(y_val, y_pred)

    r2_scores.append(r)
    mse.append(m)
    print(X_tr.shape, X_val.shape)

# Print mean and std metrics
print("=== Cross-Validation Performance ===")
print(r2_scores)
print(mse)
mean_r2_score = np.mean(r2_scores)
mean_mse = np.mean(mse)
print("%0.4f R² with a standard deviation of %0.4f" % (np.mean(r2_scores),np.std(r2_scores)))
print("%0.4f mse with a standard deviation of %0.4f" % (np.mean(mse),np.std(mse)))


# Prediction and evaluation
Y_pred = best_svr.predict(X_test).flatten()
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