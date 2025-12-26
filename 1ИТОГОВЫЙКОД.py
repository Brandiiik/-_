import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# -----------------------------
# 1) Синтетические данные
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)

n_users = 1000
n_games = 50
n_genres = 5

# Игры: genre_id (0..4), длительность(1..100), рейтинг(5..10), цена(0..60)
genre_id = np.random.randint(0, n_genres, size=n_games)
duration = np.random.uniform(1, 100, size=n_games)
rating   = np.random.uniform(5, 10,  size=n_games)
price    = np.random.uniform(0, 60,  size=n_games)

# One-hot для жанра: (n_games, 5)
genre_oh = np.eye(n_genres)[genre_id]

# Вектор признаков игры: [genre_onehot(5), duration, rating, price] => 8
games = np.column_stack([genre_oh, duration, rating, price]).astype(np.float32)  # (50, 8)

# Пользовательские предпочтения:
# веса жанров (5) + "любимая длительность/рейтинг/цена" (3) => 8
user_genre_w = np.random.uniform(0.5, 2.0, size=(n_users, n_genres)).astype(np.float32)  # (1000, 5)
user_num_pref = np.random.uniform(0.0, 1.0, size=(n_users, 3)).astype(np.float32)        # (1000, 3)
users = np.hstack([user_genre_w, user_num_pref]).astype(np.float32)                      # (1000, 8)

# -----------------------------
# 2) Генерация взаимодействий (X,y)
# -----------------------------
# "Истинный" скор: dot(users[i], games[j]) + шум
# Метка y: 1 если игра входит в top-30% для данного пользователя
X_list, y_list = [], []

for i in range(n_users):
    # scores для всех игр сразу (векторно)
    scores = users[i] @ games.T + np.random.normal(0, 0.5, size=n_games).astype(np.float32)
    thr = np.percentile(scores, 70)  # top-30%

    for j in range(n_games):
        x_ij = np.concatenate([users[i], games[j]]).astype(np.float32)  # 16 признаков
        y_ij = 1.0 if scores[j] >= thr else 0.0
        X_list.append(x_ij)
        y_list.append(y_ij)

X = np.vstack(X_list).astype(np.float32)  # (n_users*n_games, 16)
y = np.array(y_list, dtype=np.float32)    # (n_users*n_games,)

print("X shape:", X.shape, "y mean (share of 1s):", y.mean())

# -----------------------------
# 3) Train/test split + scaling
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
# StandardScaler fit/transform ожидает одинаковое число признаков (n_features) у train/test/новых данных [web:18]
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# 4) Модель (MLP)
# -----------------------------
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=256,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

# -----------------------------
# 5) Оценка качества
# -----------------------------
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(np.float32)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"Test accuracy: {acc:.4f}")
print(f"Test ROC-AUC:  {auc:.4f}")

# -----------------------------
# 6) Рекомендации для нового пользователя
# -----------------------------
# Новый пользователь: genre_weights(5) + num_pref(3)
new_user_genre_w = np.array([1.2, 0.7, 1.8, 0.9, 0.6], dtype=np.float32)
new_user_num_pref = np.array([0.8, 1.5, 0.3], dtype=np.float32)  # условные prefs
new_user = np.concatenate([new_user_genre_w,
new_user_num_pref]).astype(np.float32)  # (8,)

# Формируем входы для всех игр (n_games, 16) и нормализуем тем же scaler'ом [web:18]
X_new = np.vstack([np.concatenate([new_user, games[j]]) for j in range(n_games)]).astype(np.float32)
X_new = scaler.transform(X_new)

p_new = model.predict(X_new, verbose=0).ravel()
top5 = np.argsort(-p_new)[:5]

print("\nТоп-5 рекомендованных игр для нового пользователя:")
for rank, gid in enumerate(top5, 1):
    print(f"{rank}) Игра {int(gid)} (genre_id={int(genre_id[gid])}) -> P(recommend)={p_new[gid]:.3f}")                  
