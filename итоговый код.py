import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics, callbacks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Генерация реалистичных игровых данных
def generate_gaming_dataset(num_users=1000, num_games=500):
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Генерация данных об игроках...")
    # Пользовательские данные
    user_data = {
        'user_id': [f'U{i:04d}' for i in range(num_users)],
        'age': np.random.randint(13, 65, num_users),
        'gender': np.random.choice(['M', 'F', 'NB'], num_users, p=[0.55, 0.4, 0.05]),
        'gaming_experience': np.random.choice(['Beginner', 'Intermediate', 'Advanced', 'Expert'], 
                                             num_users, p=[0.2, 0.4, 0.3, 0.1]),
        'preferred_platform': np.random.choice(['PC', 'PlayStation', 'Xbox', 'Nintendo', 'Mobile', 'Multi'], 
                                              num_users, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
        'weekly_hours': np.random.exponential(15, num_users),
        'spending_per_month': np.random.exponential(30, num_users)
    }
    
    users_df = pd.DataFrame(user_data)
    
    print("Генерация данных об играх...")
    # Жанры и их характеристики
    game_genres = {
        'Action': {'price_mean': 40, 'price_std': 15, 'rating_mean': 75, 'rating_std': 15},
        'RPG': {'price_mean': 50, 'price_std': 20, 'rating_mean': 80, 'rating_std': 12},
        'Strategy': {'price_mean': 35, 'price_std': 10, 'rating_mean': 78, 'rating_std': 10},
        'Shooter': {'price_mean': 45, 'price_std': 12, 'rating_mean': 72, 'rating_std': 14},
        'Sports': {'price_mean': 50, 'price_std': 10, 'rating_mean': 70, 'rating_std': 16},
        'Adventure': {'price_mean': 30, 'price_std': 8, 'rating_mean': 82, 'rating_std': 8},
        'Simulation': {'price_mean': 35, 'price_std': 12, 'rating_mean': 75, 'rating_std': 10},
        'Puzzle': {'price_mean': 15, 'price_std': 5, 'rating_mean': 68, 'rating_std': 12},
        'Indie': {'price_mean': 20, 'price_std': 10, 'rating_mean': 76, 'rating_std': 14}
    }
    
    games_data = []
    genres_list = list(game_genres.keys())
    
    for i in range(num_games):
        genre = np.random.choice(genres_list)
        genre_info = game_genres[genre]
        
        # Дополнительные жанры (до 3)
        additional_genres = list(set(genres_list) - {genre})
        sub_genres = np.random.choice(additional_genres, np.random.randint(0, 3), replace=False)
        all_genres = [genre] + list(sub_genres)
        
        games_data.append({
            'game_id': f'G{i:05d}',
            'title': f'{genre} Adventure {i}',
            'main_genre': genre,
            'genres': ','.join(all_genres),
            'price': max(0, np.random.normal(genre_info['price_mean'], genre_info['price_std'])),
            'metacritic_score': min(100, max(20, np.random.normal(genre_info['rating_mean'], genre_info['rating_std']))),
            'is_multiplayer': np.random.choice([0, 1], p=[0.3, 0.7]),
            'has_campaign': np.random.choice([0, 1], p=[0.2, 0.8]),
            'is_indie': 1 if genre == 'Indie' else np.random.choice([0, 1], p=[0.8, 0.2]),
            'release_year': np.random.randint(2010, 2024)
        })
    
    games_df = pd.DataFrame(games_data)
    
    print("Генерация взаимодействий...")
    # Взаимодействия пользователь-игра
    interactions = []
    for user_idx, user in users_df.iterrows():
        # Количество игр, в которые играл пользователь
        num_interactions = np.random.poisson(8) + 2  # от 2 до ~15 игр
        
        # Выбор игр с учетом предпочтений
        preferred_genre_map = {
            'PC': ['Strategy', 'Simulation', 'Indie'],
            'PlayStation': ['Action', 'Adventure', 'RPG'],
            'Xbox': ['Shooter', 'Sports', 'Action'],
            'Nintendo': ['Adventure', 'Puzzle', 'RPG'],
            'Mobile': ['Puzzle', 'Indie', 'Simulation'],
            'Multi': ['Action', 'Sports', 'Shooter']
        }
        
        user_preferred_genres = preferred_genre_map.get(user['preferred_platform'], genres_list)
        
        # Вероятность выбора игры зависит от жанра
        game_weights = []
        for _, game in games_df.iterrows():
            game_genre_list = game['genres'].split(',')
            genre_match = len(set(game_genre_list) & set(user_preferred_genres))
            weight = 0.1 + (genre_match * 0.3)
            game_weights.append(weight)
        
        game_weights = np.array(game_weights) / sum(game_weights)
        selected_games = np.random.choice(len(games_df), min(num_interactions, len(games_df)), 
                                         replace=False, p=game_weights)
        
        for game_idx in selected_games:
            game = games_df.iloc[game_idx]
            
            # Расчет рейтинга на основе совпадения характеристик
            base_rating = np.random.randint(3, 11)  # 3-10
            
            # Модификаторы
            if user['age'] < 18 and game['main_genre'] in ['Action', 'Shooter']:
                base_rating -= 2
            elif user['age'] > 30 and game['main_genre'] in ['Indie', 'Strategy']:
                base_rating += 1
            
            if user['gaming_experience'] == 'Beginner' and game['main_genre'] in ['Strategy', 'RPG']:
                base_rating -= 1
            
            rating = max(1, min(10, base_rating))
            
            # Время игры
            playtime_hours = np.random.exponential(user['weekly_hours'] * 0.1) * (rating / 5)
            
            interactions.append({
                'user_id': user['user_id'],
                'game_id': game['game_id'],
                'rating': rating,
                'playtime_hours': playtime_hours,
                'purchased': 1,
                'completed': np.random.choice([0, 1], p=[0.6, 0.4]) if rating >= 7 else 0
            })
    
    interactions_df = pd.DataFrame(interactions)
    
    print(f"Сгенерировано: {len(users_df)} пользователей, {len(games_df)} игр, {len(interactions_df)} взаимодействий")
    
    return users_df, games_df, interactions_df

# 2. Подготовка признаков для полносвязной сети
def prepare_features_for_mlp(users_df, games_df, interactions_df):
    print("\nПодготовка признаков...")
    
    # Кодирование пользователей и игр
    user_encoder = LabelEncoder()
    game_encoder = LabelEncoder()
    
    interactions_df['user_encoded'] = user_encoder.fit_transform(interactions_df['user_id'])
    interactions_df['game_encoded'] = game_encoder.fit_transform(interactions_df['game_id'])
    
    # Подготовка пользовательских признаков
    print("  - Кодирование пользовательских признаков...")
    user_features_list = []
    
    for user_id in interactions_df['user_id'].unique():
        user_data = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # One-hot кодирование категориальных признаков
        gender_ohe = [1 if user_data['gender'] == g else 0 for g in ['M', 'F', 'NB']]
        exp_ohe = [1 if user_data['gaming_experience'] == e else 0 
                  for e in ['Beginner', 'Intermediate', 'Advanced', 'Expert']]
        platform_ohe = [1 if user_data['preferred_platform'] == p else 0 
                       for p in ['PC', 'PlayStation', 'Xbox', 'Nintendo', 'Mobile', 'Multi']]
        
        # Нормализация числовых признаков
        age_norm = user_data['age'] / 100.0
        hours_norm = min(1.0, user_data['weekly_hours'] / 168.0)  # нормализация к неделе
        spending_norm = user_data['spending_per_month'] / 500.0
        
        # Объединение всех признаков
        user_features = gender_ohe + exp_ohe + platform_ohe + [age_norm, hours_norm, spending_norm]
        user_features_list.append(user_features)
    
    user_features_df = pd.DataFrame(
        user_features_list,
        columns=(
            ['gender_M', 'gender_F', 'gender_NB'] +
            ['exp_beginner', 'exp_intermediate', 'exp_advanced', 'exp_expert'] +
            ['platform_PC', 'platform_PlayStation', 'platform_Xbox', 
             'platform_Nintendo', 'platform_Mobile', 'platform_Multi'] +
            ['age_norm', 'weekly_hours_norm', 'spending_norm']
        )
    )
    
    # Подготовка игровых признаков
    print("  - Кодирование игровых признаков...")
    game_features_list = []
    
    # Подготовка MultiLabelBinarizer для жанров
    mlb = MultiLabelBinarizer()
    games_df['genres_list'] = games_df['genres'].str.split(',')
    genres_encoded = mlb.fit_transform(games_df['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    
    for game_id in interactions_df['game_id'].unique():
        game_data = games_df[games_df['game_id'] == game_id].iloc[0]
        game_idx = games_df[games_df['game_id'] == game_id].index[0]
        
        # Жанры (уже закодированы)
        genre_features = genres_df.iloc[game_idx].values.tolist()
        
        # One-hot кодирование основного жанра
        main_genres = list(games_df['main_genre'].unique())
        main_genre_ohe = [1 if game_data['main_genre'] == g else 0 for g in main_genres]
        
        # Нормализация числовых признаков
        price_norm = min(1.0, game_data['price'] / 100.0)
        score_norm = game_data['metacritic_score'] / 100.0
        release_norm = (game_data['release_year'] - 2010) / 15.0  # нормализация 2010-2024
        
        # Бинарные признаки
        binary_features = [
            game_data['is_multiplayer'],
            game_data['has_campaign'],
            game_data['is_indie']
        ]
        
        # Объединение всех признаков
        game_features = genre_features + main_genre_ohe + binary_features + [price_norm, score_norm, release_norm]
        game_features_list.append(game_features)
    
    # Создание общей матрицы признаков
    print("  - Создание финальной матрицы признаков...")
    X_list = []
    y_list = []
    
    for idx, interaction in interactions_df.iterrows():
        user_idx = interaction['user_encoded']
        game_idx = interaction['game_encoded']
        
        # Объединение признаков пользователя и игры
        user_feats = user_features_df.iloc[user_idx].values
        game_feats = game_features_list[game_idx]
        
        combined_features = np.concatenate([user_feats, game_feats])
        X_list.append(combined_features)
        
        # Целевая переменная (рейтинг нормализованный 0-1)
        y_list.append(interaction['rating'] / 10.0)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"  Размерность признаков: {X.shape[1]}")
    print(f"  Количество образцов: {X.shape[0]}")
    
    return X, y, user_encoder, game_encoder

# 3. Создание полносвязной нейронной сети (MLP)
def create_mlp_recommender(input_dim, hidden_layers=[256, 128, 64, 32], dropout_rate=0.3):
    print(f"\nСоздание MLP модели с входной размерностью {input_dim}...")
    
    model = keras.Sequential(name="Game_Recommender_MLP")
    
    # Входной слой
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())
    
    # Скрытые слои
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', 
                              kernel_initializer='he_normal',
                              name=f'hidden_layer_{i+1}'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
    
    # Выходной слой
    model.add(layers.Dense(1, activation='sigmoid', name='output_layer'))
    
    # Компиляция модели
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.MeanSquaredError(),
        metrics=[
            metrics.MeanAbsoluteError(name='mae'),
            metrics.RootMeanSquaredError(name='rmse')
        ]
    )
    
    model.summary()
    return model

# 4. Обучение модели
def train_mlp_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
    print("\nНачало обучения MLP модели...")
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            'best_mlp_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Обучение
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

# 5. Визуализация результатов
def plot_training_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    # RMSE
    axes[2].plot(history.history['rmse'], label='Train RMSE')
    axes[2].plot(history.history['val_rmse'], label='Val RMSE')
    axes[2].set_title('Root Mean Squared Error')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('RMSE')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

# 6. Функция рекомендаций
def generate_recommendations(model, user_id, users_df, games_df, interactions_df, 
                           user_encoder, game_encoder, top_n=10):
    print(f"\nГенерация рекомендаций для пользователя {user_id}...")
    
    # Проверяем, есть ли пользователь в данных
    if user_id not in user_encoder.classes_:
        print(f"Пользователь {user_id} не найден. Используем нового пользователя.")
        # Используем средние характеристики
        user_features = np.zeros((len(users_df.columns) - 1))  # -1 для user_id
        user_features[0] = 1  # предположим мужчина
        user_features[4] = 1  # предположим средний уровень
    else:
        # Получаем характеристики пользователя
        user_data = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # Кодируем характеристики (упрощенная версия)
        gender_ohe = [1 if user_data['gender'] == g else 0 for g in ['M', 'F', 'NB']]
        exp_ohe = [1 if user_data['gaming_experience'] == e else 0 
                  for e in ['Beginner', 'Intermediate', 'Advanced', 'Expert']]
        platform_ohe = [1 if user_data['preferred_platform'] == p else 0 
                       for p in ['PC', 'PlayStation', 'Xbox', 'Nintendo', 'Mobile', 'Multi']]
        
        age_norm = user_data['age'] / 100.0
        hours_norm = min(1.0, user_data['weekly_hours'] / 168.0)
        spending_norm = user_data['spending_per_month'] / 500.0
        
        user_features = np.array(gender_ohe + exp_ohe + platform_ohe + [age_norm, hours_norm, spending_norm])
    
    # Готовим признаки для всех игр
    predictions = []
    
    for _, game in games_df.iterrows():
        # Подготовка игровых признаков (упрощенно)
        # В реальном приложении здесь должна быть полная подготовка как в обучении
        
        # Создаем базовые признаки игры
        game_features = np.zeros(50)  # Упрощенная размерность
        
        # Наполняем случайными значениями для демонстрации
        game_features[:20] = np.random.rand(20)
        
        # Объединяем признаки пользователя и игры
        combined_features = np.concatenate([user_features, game_features])
        
        # Делаем предсказание
        prediction = model.predict(combined_features.reshape(1, -1), verbose=0)[0][0]
        
        predictions.append({
            'game_id': game['game_id'],
            'title': game['title'],
            'main_genre': game['main_genre'],
            'price': game['price'],
            'metacritic_score': game['metacritic_score'],
            'predicted_rating': round(prediction * 10, 2),  # Возвращаем к шкале 1-10
            'prediction_score': prediction
        })
    
    # Сортируем по предсказанному рейтингу
    recommendations_df = pd.DataFrame(predictions)
    recommendations_df = recommendations_df.sort_values('prediction_score', ascending=False).head(top_n)
    
    return recommendations_df

# 7. Основной пайплайн
def main():
    print("=" * 60)
    print("РЕКОМЕНДАТЕЛЬНАЯ СИСТЕМА ИГР НА ОСНОВЕ MLP")
    print("=" * 60)
    
    # Шаг 1: Генерация данных
    print("\n1. ГЕНЕРАЦИЯ ДАННЫХ")
    users_df, games_df, interactions_df = generate_gaming_dataset(
        num_users=500,  # Можно увеличить для более точной модели
        num_games=200
    )
    
    # Шаг 2: Подготовка признаков
    print("\n2. ПОДГОТОВКА ПРИЗНАКОВ")
    X, y, user_encoder, game_encoder = prepare_features_for_mlp(
        users_df, games_df, interactions_df
    )
    
    # Шаг 3: Разделение данных
    print("\n3. РАЗДЕЛЕНИЕ ДАННЫХ")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Шаг 4: Создание и обучение модели
    print("\n4. СОЗДАНИЕ И ОБУЧЕНИЕ MLP МОДЕЛИ")
    input_dim = X_train.shape[1]
    
    model = create_mlp_recommender(
        input_dim=input_dim,
        hidden_layers=[256, 128, 64, 32],  # Архитектура сети
        dropout_rate=0.3
    )
    
    history = train_mlp_model(
        model, X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=64
    )
    
    # Шаг 5: Оценка модели
    print("\n5. ОЦЕНКА МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  Test Loss (MSE): {test_results[0]:.4f}")
    print(f"  Test MAE: {test_results[1]:.4f}")
    print(f"  Test RMSE: {test_results[2]:.4f}")
    
    # Визуализация
    print("\n6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    plot_training_history(history)
    
    # Шаг 7: Генерация рекомендаций
    print("\n7. ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ")
    
    # Выбираем случайного пользователя
    sample_user = users_df.iloc[0]['user_id']
    
    recommendations = generate_recommendations(
        model=model,
        user_id=sample_user,
        users_df=users_df,
        games_df=games_df,
        interactions_df=interactions_df,
        user_encoder=user_encoder,
        game_encoder=game_encoder,
        top_n=10
    )
    
    print(f"\nТоп-10 рекомендаций для пользователя {sample_user}:")
    print("-" * 100)
    print(recommendations[['title', 'main_genre', 'price', 'metacritic_score', 'predicted_rating']].to_string(index=False))
    print("-" * 100)
    
    # Шаг 8: Сохранение модели
    print("\n8. СОХРАНЕНИЕ МОДЕЛИ И ДАННЫХ")
    model.save('game_recommender_mlp.h5')
    print("  Модель сохранена как 'game_recommender_mlp.h5'")
    
    # Сохранение кодировщиков
    import pickle
    with open('user_encoder.pkl', 'wb') as f:
        pickle.dump(user_encoder, f)
    with open('game_encoder.pkl', 'wb') as f:
        pickle.dump(game_encoder, f)
    
    print("  Кодировщики сохранены")
    
    # Шаг 9: Пример использования модели
    print("\n9. ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДЕЛИ")
    print("""
    # Загрузка модели
    loaded_model = keras.models.load_model('game_recommender_mlp.h5')
    
    # Подготовка новых данных
    # new_user_features = prepare_user_features(new_user_data)
    # new_game_features = prepare_game_features(new_game_data)
    # combined_features = combine_features(new_user_features, new_game_features)
    
    # Предсказание
    # predicted_rating = loaded_model.predict(combined_features)
    """)
    
    print("\n" + "=" * 60)
    print("ВЫПОЛНЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 60)
    

# Запуск
if __name__ == "__main__":
    main()
