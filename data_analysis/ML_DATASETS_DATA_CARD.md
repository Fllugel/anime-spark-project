# Data Card: Anime ML Datasets

## Overview

Цей репозиторій містить два підготовлених датасети для машинного навчання на основі даних про аніме з MyAnimeList (MAL):

1. **Regression Dataset** - для передбачення рейтингу (Score) аніме
2. **Classification Dataset** - для передбачення статі користувача за його anime preferences

---

## Dataset 1: Regression (Передбачення Score аніме)

### Опис задачі
Регресійна модель для передбачення середньої оцінки (Score) аніме на основі його характеристик.

### Розташування
```
data/ml_datasets/regression/
├── train.csv / train.parquet          # Тренувальна вибірка
├── validation.csv / validation.parquet # Валідаційна вибірка  
├── test.csv / test.parquet            # Тестова вибірка
├── preprocessing_info.json            # Метадані про features
└── split_info.json                    # Інформація про розбиття
```

### Статистика
| Метрика | Значення |
|---------|----------|
| Загальна кількість записів | 14,948 |
| Train | 10,580 (70.8%) |
| Validation | 2,170 (14.5%) |
| Test | 2,198 (14.7%) |
| Кількість features | 79 |
| Target | Score (float, 1-10) |

### Джерело даних
- **Файл**: `anime-filtered.csv`
- **Опис**: Відфільтрований датасет з інформацією про аніме з MyAnimeList

### Features

#### Числові features (нормалізовані min-max):
| Feature | Опис |
|---------|------|
| `episodes_clean` | Кількість епізодів |
| `duration_minutes` | Тривалість епізоду в хвилинах |
| `release_year` | Рік випуску |
| `members_normalized` | Кількість користувачів у списках (нормалізовано) |
| `favorites_normalized` | Кількість у "Улюблених" (нормалізовано) |
| `popularity_normalized` | Ранг популярності (нормалізовано) |
| `watching_normalized` | Кількість тих, хто дивиться (нормалізовано) |
| `completed_normalized` | Кількість тих, хто завершив (нормалізовано) |

#### Categorical features (one-hot encoded):

**Жанри (30 категорій):**
Action, Adventure, Comedy, Drama, Fantasy, Romance, Sci-Fi, Slice of Life, Sports, Supernatural, Mystery, Horror, Psychological, Thriller, Mecha, Music, School, Shounen, Shoujo, Seinen, Josei, Ecchi, Harem, Isekai, Military, Historical, Demons, Magic, Super Power, Vampire

**Типи (6 категорій):**
TV, Movie, OVA, ONA, Special, Music

**Джерела (9 категорій):**
Manga, Original, Light novel, Visual novel, Game, Novel, Web manga, 4-koma manga, Other

**Вікові рейтинги (6 категорій):**
G - All Ages, PG - Children, PG-13, R - 17+, R+, Rx - Hentai

**Топ студії (20 категорій):**
Toei Animation, Madhouse, Sunrise, J.C.Staff, A-1 Pictures, Bones, Production I.G, Studio Pierrot, MAPPA, Kyoto Animation, ufotable, Wit Studio, CloverWorks, Studio Ghibli, Shaft, TMS Entertainment, OLM, Brain's Base, Doga Kobo, P.A. Works

### Процес підготовки

1. **Завантаження**: Завантажено `anime-filtered.csv` через PySpark
2. **Фільтрація**: Видалено записи з відсутнім або нульовим Score
3. **Feature extraction**:
   - Витягнуто duration в хвилинах з текстового поля
   - Витягнуто рік випуску з поля Aired
4. **Encoding**:
   - One-hot encoding для жанрів (multi-label)
   - One-hot encoding для Type, Source, Rating
   - Binary encoding для топ-20 студій
5. **Нормалізація**: Min-max scaling для числових features
6. **Split**: Random split 70/15/15

---

## Dataset 2: Classification (Передбачення Gender користувача)

### Опис задачі
Класифікаційна модель для передбачення статі користувача (Male/Female) на основі його anime preferences та поведінки на платформі.

### Розташування
```
data/ml_datasets/classification/
├── train.csv / train.parquet          # Тренувальна вибірка
├── validation.csv / validation.parquet # Валідаційна вибірка
├── test.csv / test.parquet            # Тестова вибірка
├── preprocessing_info.json            # Метадані про features
└── split_info.json                    # Інформація про розбиття
```

### Статистика
| Метрика | Значення |
|---------|----------|
| Загальна кількість записів | 10,507 |
| Train | 7,354 (70.0%) |
| Validation | 1,576 (15.0%) |
| Test | 1,577 (15.0%) |
| Кількість features | 80 |
| Target | gender_encoded (0=Male, 1=Female) |

### Баланс класів
| Клас | Кількість | Відсоток |
|------|-----------|----------|
| Male | 6,328 | 60.2% |
| Female | 4,179 | 39.8% |

### Джерела даних
- **users-details-2023.csv**: Демографічні дані та статистика користувачів
- **users-score-2023.csv**: Оцінки користувачів для аніме
- **anime-filtered.csv**: Інформація про аніме (для агрегації жанрів)

### Features

#### Статистика оцінок користувача:
| Feature | Опис |
|---------|------|
| `total_ratings` | Загальна кількість оцінок |
| `avg_rating` | Середня оцінка користувача |
| `rating_std` | Стандартне відхилення оцінок |
| `min_rating` / `max_rating` | Мін/макс оцінка |
| `unique_anime_count` | Кількість унікальних аніме |

#### Жанрові preferences (23 жанри):
- `genre_count_*` - кількість переглянутих аніме кожного жанру
- `genre_pct_*` - відсоток переглядів кожного жанру

#### Preferences за типами аніме:
- `type_count_*` - кількість переглядів за типом (TV, Movie, OVA, ONA, Special)
- `avg_rating_*` - середня оцінка за типом

#### Демографічні дані користувача:
| Feature | Опис |
|---------|------|
| `days_watched` | Загальна кількість днів перегляду |
| `user_mean_score` | Середня оцінка з профілю |
| `user_completed` | Кількість завершених аніме |
| `user_dropped` | Кількість кинутих аніме |
| `user_plan_to_watch` | Кількість в планах |
| `user_total_entries` | Загальна кількість записів |
| `user_rewatched` | Кількість повторних переглядів |
| `user_episodes_watched` | Загальна кількість епізодів |

#### Derived features:
| Feature | Опис |
|---------|------|
| `genre_diversity` | Кількість різних жанрів (> 0 переглядів) |
| `rating_consistency` | rating_std / avg_rating |
| `preference_intensity` | (max_rating - min_rating) / 9 |
| `drop_rate` | user_dropped / user_total_entries |

### Процес підготовки

1. **Завантаження користувачів**: 
   - Завантажено `users-details-2023.csv`
   - Відфільтровано користувачів з відомою статтю (Male/Female)
   - Random sampling: 20,000 користувачів

2. **Завантаження оцінок**:
   - Читання `users-score-2023.csv` частинами (chunks по 500k рядків)
   - Фільтрація на льоту тільки для відібраних користувачів
   - Це дозволило ефективно обробити файл 1.1GB

3. **Агрегація features**:
   - Join ratings з anime info
   - GroupBy по user_id з розрахунком статистик
   - Агрегація жанрів та типів

4. **Feature engineering**:
   - Створення derived features
   - Genre percentages
   - Min-max нормалізація

5. **Фільтрація**: Видалено користувачів з < 10 оцінками

6. **Split**: Stratified split 70/15/15 за gender_encoded

---

## Використання

### Завантаження датасетів (Python)

```python
import pandas as pd

# Regression dataset
train_reg = pd.read_csv('data/ml_datasets/regression/train.csv')
# або
train_reg = pd.read_parquet('data/ml_datasets/regression/train.parquet')

# Classification dataset
train_clf = pd.read_csv('data/ml_datasets/classification/train.csv')
# або
train_clf = pd.read_parquet('data/ml_datasets/classification/train.parquet')
```

### Приклад моделі регресії

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Завантаження
train = pd.read_parquet('data/ml_datasets/regression/train.parquet')
test = pd.read_parquet('data/ml_datasets/regression/test.parquet')

# Features та target
feature_cols = [c for c in train.columns if c not in ['anime_id', 'Name', 'Score']]
X_train, y_train = train[feature_cols], train['Score']
X_test, y_test = test[feature_cols], test['Score']

# Модель
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцінка
y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
```

### Приклад моделі класифікації

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Завантаження
train = pd.read_parquet('data/ml_datasets/classification/train.parquet')
test = pd.read_parquet('data/ml_datasets/classification/test.parquet')

# Features та target
feature_cols = [c for c in train.columns if c not in ['user_id', 'Gender', 'gender_encoded']]
X_train, y_train = train[feature_cols], train['gender_encoded']
X_test, y_test = test[feature_cols], test['gender_encoded']

# Модель
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оцінка
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))
```

---

## Обмеження та етичні міркування

### Data Limitations
- Дані обмежені платформою MyAnimeList і можуть не представляти всю anime-аудиторію
- Sampling 20,000 користувачів може не відображати всі паттерни
- Стать визначається користувачами самостійно і може бути неточною

### Ethical Considerations
- Датасет класифікації статі слід використовувати тільки для дослідницьких цілей
- Не рекомендується використовувати для персоналізації без згоди користувача
- Результати можуть відображати стереотипи, присутні в даних

---

## Ліцензія та Attribution

Оригінальні дані отримано з MyAnimeList Dataset 2023.

---

## Автор

**Bohdan Osmuk**

Дата створення: December 2024

