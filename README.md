### Вариант 1 для Евгения Володина

#### Программа на Python для MVP (прогнозирование курса валют)Давай напишем код для MVP, используя модель ARIMA, так как она хорошо подходит для временных рядов. 
Если ARIMA покажется сложной или данные будут плохо подходить, можно упростить до линейной регрессии на лаговых признаках. 
#### Предположения по данным из файла RC_F01_05_2020_T22_05_2025.xlsx:

#### Данные находятся на листе с именем "RC".
Столбец 'data' содержит даты, но они в формате чисел Excel (количество дней с 01.01.1900 или 01.01.1904). Их нужно будет корректно преобразовать.
Столбец 'curs' – это курс.
Столбец 'cdx' указывает, что это "Доллар США". Мы будем фильтровать по нему, хотя в данном файле, похоже, только он и есть.

#### Шаг 1: Установка необходимых библиотек
Если они еще не установлены:

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn openpyxl
```

#### сама прога

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Для моделей временных рядов
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Для оценки
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Отключим предупреждения (опционально, для чистоты вывода)
import warnings
warnings.filterwarnings("ignore")

# --- 1. Загрузка и предварительная обработка данных ---
print("--- 1. Загрузка и предварительная обработка данных ---")
try:
    # Укажите правильный путь к вашему файлу Excel
    excel_file_path = 'RC_F01_05_2020_T22_05_2025.xlsx'
    df = pd.read_excel(excel_file_path, sheet_name='RC')
    print("Данные успешно загружены.")
    print("Первые 5 строк исходных данных:")
    print(df.head())
except FileNotFoundError:
    print(f"Ошибка: Файл {excel_file_path} не найден.")
    exit()
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    exit()

# Выбор нужных столбцов и переименование для удобства
df_processed = df[['data', 'curs', 'cdx']].copy()
df_processed.rename(columns={'data': 'DateExcel', 'curs': 'Rate', 'cdx': 'Currency'}, inplace=True)

# Фильтрация данных (на случай, если есть другие валюты, хотя в примере только доллар)
df_processed = df_processed[df_processed['Currency'] == 'Доллар США']
if df_processed.empty:
    print("Ошибка: Данные по Доллару США не найдены.")
    exit()

# Преобразование дат Excel в datetime
# Число дней с 30.12.1899 (стандарт Excel для Windows)
# Если даты отображаются некорректно, возможно, нужно использовать другую базовую дату
# или проверить, как именно Excel хранит эти числа (иногда как строки)
def excel_date_to_datetime(excel_date):
    if isinstance(excel_date, (int, float)):
        return pd.Timestamp('1899-12-30') + pd.to_timedelta(excel_date, 'D')
    try:
        # Попытка прямого преобразования, если это уже строка с датой
        return pd.to_datetime(excel_date)
    except ValueError:
        return pd.NaT # Возвращаем NaT, если не можем преобразовать

df_processed['Date'] = df_processed['DateExcel'].apply(excel_date_to_datetime)

# Удаление строк с некорректными датами
df_processed.dropna(subset=['Date'], inplace=True)
if df_processed.empty:
    print("Ошибка: Не удалось преобразовать даты или все даты некорректны.")
    exit()

# Преобразование курса в числовой формат (замена запятых на точки, если есть)
if df_processed['Rate'].dtype == 'object':
    df_processed['Rate'] = df_processed['Rate'].astype(str).str.replace(',', '.').astype(float)
else:
    df_processed['Rate'] = df_processed['Rate'].astype(float)


# Установка даты как индекса и сортировка
df_processed.set_index('Date', inplace=True)
df_processed.sort_index(inplace=True)

# Удаление ненужных столбцов
df_processed = df_processed[['Rate']]

print("\nОбработанные данные (первые 5 строк):")
print(df_processed.head())
print(f"\nВсего записей после обработки: {len(df_processed)}")
print(f"Временной диапазон: от {df_processed.index.min()} до {df_processed.index.max()}")

# Проверка на пропуски
if df_processed['Rate'].isnull().sum() > 0:
    print(f"\nОбнаружены пропуски в данных: {df_processed['Rate'].isnull().sum()} шт.")
    # Простая стратегия заполнения пропусков - предыдущим значением
    df_processed['Rate'].fillna(method='ffill', inplace=True)
    # Или можно удалить строки с пропусками, если их мало: df_processed.dropna(inplace=True)
    print("Пропуски заполнены методом ffill.")

# --- 2. Разведочный анализ данных (EDA) и проверка стационарности ---
print("\n--- 2. Разведочный анализ данных (EDA) ---")
plt.figure(figsize=(12, 6))
plt.plot(df_processed['Rate'], label='Курс Доллара США')
plt.title('Исторический курс Доллара США к Рублю')
plt.xlabel('Дата')
plt.ylabel('Курс')
plt.legend()
plt.grid(True)
plt.show()

# Проверка на стационарность с помощью теста Дики-Фуллера
print("\nПроверка ряда на стационарность (тест Дики-Фуллера):")
result_adf = adfuller(df_processed['Rate'])
print(f'ADF Statistic: {result_adf[0]}')
print(f'p-value: {result_adf[1]}')
print('Critical Values:')
for key, value in result_adf[4].items():
    print(f'\t{key}: {value}')

if result_adf[1] > 0.05:
    print("Ряд НЕ является стационарным (p-value > 0.05). Требуется дифференцирование.")
    # Попробуем взять первую разность для достижения стационарности
    df_processed['Rate_diff'] = df_processed['Rate'].diff().dropna()
    
    result_adf_diff = adfuller(df_processed['Rate_diff'])
    print("\nТест Дики-Фуллера для ряда первых разностей:")
    print(f'ADF Statistic: {result_adf_diff[0]}')
    print(f'p-value: {result_adf_diff[1]}')
    if result_adf_diff[1] <= 0.05:
        print("Ряд первых разностей является стационарным.")
        # d=1 в ARIMA
        time_series_data = df_processed['Rate_diff']
        d_order = 1
    else:
        print("Ряд первых разностей все еще не стационарен. Для MVP ARIMA может не подойти или требует дальнейшей обработки.")
        print("Для MVP можно попробовать использовать исходный ряд, осознавая ограничения.")
        time_series_data = df_processed['Rate'] # Используем исходный ряд для простоты MVP
        d_order = 0 # Или попробовать d=1, но модель может быть нестабильной
else:
    print("Ряд является стационарным (p-value <= 0.05). d=0 в ARIMA.")
    time_series_data = df_processed['Rate']
    d_order = 0

# Графики ACF и PACF для подбора порядков p и q для ARIMA
# Строим для ряда, который будем моделировать (стационарного или исходного)
if not time_series_data.empty:
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(time_series_data, ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    plt.subplot(122)
    plot_pacf(time_series_data, ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.show()
else:
    print("Недостаточно данных для построения ACF/PACF после дифференцирования.")
    # В таком случае, для MVP можно выбрать p и q эвристически (например, 1 или 2)

# --- 3. Подготовка данных и выбор модели ---
print("\n--- 3. Подготовка данных и выбор модели ---")
# Разделение данных на обучающую и тестовую выборки
# Возьмем последние N дней для теста, например, 60 дней
train_size_ratio = 0.9 # или фиксированное количество N дней
if len(time_series_data) > 60 : # Убедимся, что данных достаточно
    split_point = int(len(time_series_data) * train_size_ratio)
    train_data = time_series_data[:split_point]
    test_data = time_series_data[split_point:]
    
    # Для восстановления прогнозов, если работали с разностями
    last_obs_train_orig_rate = df_processed['Rate'].iloc[split_point-1] if d_order > 0 else None

    print(f"Размер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")
else:
    print("Недостаточно данных для разделения на обучение и тест. Модель не будет обучена.")
    train_data = pd.Series() # Пустые серии
    test_data = pd.Series()

# --- 4. Обучение модели ARIMA ---
# Для MVP подберем параметры p, d, q эвристически, например (5,d,2) или (2,d,1)
# d уже определили выше
# p - порядок AR (смотрим на PACF)
# q - порядок MA (смотрим на ACF)
# Пусть для примера p=5, q=2. В реальной задаче их нужно подбирать тщательнее.
p_order = 5
q_order = 2

model_arima = None
if not train_data.empty:
    print(f"\n--- 4. Обучение модели ARIMA({p_order},{d_order},{q_order}) ---")
    try:
        # Если используем исходный нестационарный ряд (d=0), модель будет обучаться на нем
        # Если использовали дифференцированный (d=1), то train_data уже является им
        model_arima = ARIMA(train_data, order=(p_order, d_order, q_order))
        model_fit_arima = model_arima.fit()
        print(model_fit_arima.summary())
        print("Модель ARIMA успешно обучена.")
    except Exception as e:
        print(f"Ошибка при обучении модели ARIMA: {e}")
        model_arima = None # Сброс, если обучение не удалось
else:
    print("Обучающая выборка пуста, обучение модели ARIMA невозможно.")


# --- 5. Прогнозирование и оценка модели ---
arima_predictions = []
if model_arima is not None and model_fit_arima is not None and not test_data.empty:
    print("\n--- 5. Прогнозирование и оценка модели ARIMA ---")
    # Прогнозирование на тестовой выборке
    start_index_pred = len(train_data)
    end_index_pred = len(train_data) + len(test_data) - 1
    
    # `predict` для ARIMA в statsmodels может работать по-разному в зависимости от версии
    # Обычно start и end - это индексы относительно начала полного ряда (train+test)
    # или индексы от начала обучающего ряда
    
    # Попробуем сделать прогноз "шаг за шагом" или одним блоком
    try:
        # Прогноз на количество шагов, равное длине тестовой выборки
        forecast_steps = len(test_data)
        arima_forecast_result = model_fit_arima.get_forecast(steps=forecast_steps)
        arima_predictions_diff = arima_forecast_result.predicted_mean # Это прогноз для дифференцированного ряда, если d > 0
        
        # Преобразование прогнозов обратно, если d > 0
        if d_order > 0:
            # Восстановление прогнозов. Нам нужны фактические значения курса, а не их разности.
            # Это сложнее для многошагового прогноза разностей.
            # Простой вариант: предполагаем, что у нас есть последний фактический курс из обучающей выборки,
            # и к нему кумулятивно прибавляем предсказанные разности.
            
            # Более корректный путь - использовать `predict` с dynamic=False для одношаговых,
            # или аккуратно восстанавливать из прогноза разностей.
            # Для MVP, если использовали d_order=1:
            history = list(train_data.values) # или df_processed['Rate'][:split_point].values для исходных курсов
            predictions_restored = []
            
            # Восстанавливаем прогноз для исходного ряда
            # Если d=1, то Rate_t = Rate_{t-1} + Diff_t
            # Прогноз для Diff_t у нас есть в arima_predictions_diff

            # Нужен последний известный курс из обучающей выборки
            # df_processed['Rate'][split_point-1]
            last_actual_rate_in_train = df_processed['Rate_diff'].index[split_point-1] # Дата последнего значения в трейне
            last_actual_rate_value = df_processed.loc[last_actual_rate_in_train, 'Rate']

            current_rate = last_actual_rate_value
            for forecasted_diff in arima_predictions_diff:
                current_rate = current_rate + forecasted_diff
                predictions_restored.append(current_rate)
            arima_predictions = pd.Series(predictions_restored, index=test_data.index)

        else: # d_order == 0
            arima_predictions = arima_predictions_diff # Прогноз уже для исходного ряда
            arima_predictions.index = test_data.index # Присваиваем корректный индекс


        # Оценка качества
        actual_test_values = df_processed['Rate'][split_point:] # Исходные курсы на тестовом периоде
        
        if len(arima_predictions) == len(actual_test_values):
            rmse_arima = np.sqrt(mean_squared_error(actual_test_values, arima_predictions))
            mae_arima = mean_absolute_error(actual_test_values, arima_predictions)
            print(f'ARIMA RMSE: {rmse_arima:.4f}')
            print(f'ARIMA MAE: {mae_arima:.4f}')

            # Визуализация результатов
            plt.figure(figsize=(14, 7))
            plt.plot(df_processed.index, df_processed['Rate'], label='Исторический курс (Весь)', color='blue')
            plt.plot(train_data.index, df_processed['Rate'][:split_point], label='Обучающие данные (Исходный курс)', color='green') # Показываем исходный курс
            plt.plot(actual_test_values.index, actual_test_values, label='Тестовые данные (Факт)', color='orange')
            plt.plot(arima_predictions.index, arima_predictions, label='Прогноз ARIMA', color='red', linestyle='--')
            plt.title('Прогноз курса Доллара США моделью ARIMA')
            plt.xlabel('Дата')
            plt.ylabel('Курс')
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("Размеры прогноза и фактических тестовых данных не совпадают. Оценка невозможна.")
            print(f"Длина прогноза: {len(arima_predictions)}, Длина теста: {len(actual_test_values)}")


    except Exception as e:
        print(f"Ошибка при прогнозировании или оценке: {e}")
        if arima_predictions_diff is not None: # Если прогноз разностей был получен
             arima_predictions = arima_predictions_diff
             arima_predictions.index = test_data.index
             print("Прогноз (возможно, для дифференцированного ряда):")
             print(arima_predictions.head())
        else:
            print("Прогноз не был сгенерирован.")


# --- 6. Прогноз на будущее (несколько дней вперед от конца имеющихся данных) ---
if model_arima is not None and model_fit_arima is not None:
    print("\n--- 6. Прогноз на несколько дней в будущее ---")
    # Обучим модель на всех доступных данных (из `time_series_data`)
    # Это `time_series_data` может быть исходным или дифференцированным рядом
    try:
        full_model_arima = ARIMA(time_series_data, order=(p_order, d_order, q_order))
        full_model_fit_arima = full_model_arima.fit()
        print("Модель ARIMA успешно переобучена на всех данных.")

        future_steps = 7 # Прогноз на 7 дней вперед
        future_forecast_result = full_model_fit_arima.get_forecast(steps=future_steps)
        future_predicted_means_diff = future_forecast_result.predicted_mean
        future_conf_int = future_forecast_result.conf_int()

        # Восстановление прогноза, если d > 0
        last_date_in_data = df_processed.index[-1]
        future_dates = pd.date_range(start=last_date_in_data + timedelta(days=1), periods=future_steps, freq='B') # Используем 'B' - business day frequency

        if d_order > 0:
            last_actual_rate_full_data = df_processed['Rate'][-1]
            current_rate_future = last_actual_rate_full_data
            future_predictions_restored = []
            for forecasted_diff_future in future_predicted_means_diff:
                current_rate_future = current_rate_future + forecasted_diff_future
                future_predictions_restored.append(current_rate_future)
            future_forecast_values = pd.Series(future_predictions_restored, index=future_dates)
             # Доверительные интервалы для разностей, их восстановление сложнее, для MVP опустим
        else:
            future_forecast_values = pd.Series(future_predicted_means_diff.values, index=future_dates)
            # Для d=0, можно попробовать отобразить доверительные интервалы
            future_conf_int.index = future_dates


        print("\nПрогноз на ближайшие дни:")
        print(future_forecast_values)

        plt.figure(figsize=(12, 6))
        plt.plot(df_processed['Rate'][-100:], label='История (последние 100 дней)') # Показать часть истории
        plt.plot(future_forecast_values, label='Прогноз на будущее', color='red', linestyle='--')
        if d_order == 0 and future_conf_int is not None: # Показать доверительные интервалы, если d=0
             plt.fill_between(future_conf_int.index,
                             future_conf_int.iloc[:, 0],
                             future_conf_int.iloc[:, 1], color='pink', alpha=0.3, label='Доверительный интервал 95%')
        plt.title('Прогноз курса Доллара США на ближайшие дни')
        plt.xlabel('Дата')
        plt.ylabel('Курс')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Ошибка при прогнозировании на будущее: {e}")

print("\n--- Завершение работы MVP ---")

```
#### Загрузка данных и предобработка:

#### Код загружает данные из Excel. Важно правильно обработать столбец с датами, так как Excel хранит их как числа. Функция excel_date_to_datetime пытается это сделать. Если даты в файле уже в текстовом формате, pd.to_datetime справится.
Курс валюты преобразуется в числовой тип (на случай, если там десятичные разделители в виде запятой).
Пропуски заполняются методом ffill (предыдущим значением) – это простая стратегия для MVP. В полноценном исследовании можно рассмотреть другие методы.

#### Разведочный анализ (EDA):

#### Строится график временного ряда.
Проводится тест Дики-Фуллера на стационарность. Если ряд нестационарен (p-value > 0.05), берется первая разность, и тест повторяется. Это важно для ARIMA, так как она предполагает стационарность ряда (или ее компонент d учитывает порядок интегрирования).
Строятся автокорреляционная (ACF) и частично автокорреляционная (PACF) функции. Они помогают подобрать параметры p (по PACF) и q (по ACF) для модели ARIMA. Для MVP параметры (p,d,q) можно выбрать эвристически (например, (5,1,2) или (2,1,1) если ряд дифференцировался один раз, d=1).

#### Модель ARIMA (statsmodels.tsa.arima.model.ARIMA):

order=(p, d, q):

p: порядок авторегрессионной части (AR).
d: порядок интегрирования (количество взятых разностей для достижения стационарности).
q: порядок части скользящего среднего (MA).

#### Модель обучается на train_data.
Выводится summary() модели, где можно посмотреть коэффициенты, их значимость и другую статистику.

Прогнозирование и оценка:

Делается прогноз на тестовой выборке.
Важный момент: если модель обучалась на дифференцированных данных (d>0), то и прогнозы она выдаст для дифференцированного ряда. Их нужно будет "восстановить" до исходных значений курса. В коде предпринята попытка это сделать наиболее простым образом для многошагового прогноза. Корректное восстановление для многошаговых прогнозов разностей может быть сложным, так как ошибка имеет свойство накапливаться.
Метрики: RMSE и MAE.
Визуализация: Сравнение фактических и предсказанных значений.

Прогноз на будущее:

Модель переобучается на всех доступных исторических данных.
Делается прогноз на несколько дней вперед.
Также выполняется попытка восстановить прогноз к исходным значениям, если d>0.

Что нужно будет описать в дипломе по этому коду:

Каждый шаг из кода (загрузка, предобработка, EDA, выбор и обучение модели, оценка, прогноз).
Обоснование выбора параметров p, d, q для ARIMA (можно сослаться на графики ACF/PACF или указать, что для MVP они выбраны эвристически).
Результаты теста Дики-Фуллера.
Полученные метрики качества (RMSE, MAE) и их интерпретация.
Графики, построенные программой.
Обязательно указать ограничения MVP:

Модель ARIMA учитывает только прошлые значения самого ряда и не использует внешние факторы (новости, экономические индикаторы, политические события и т.д.), которые сильно влияют на курс.
Подбор параметров p,q для ARIMA в данном MVP упрощен.
Восстановление прогноза из разностей для многошаговых горизонтов может вносить дополнительные погрешности.
Финансовые рынки очень волатильны и труднопредсказуемы, особенно на коротких временных интервалах.

#### Пути улучшения MVP:

Использование более сложных моделей (SARIMA для учета сезонности, если она есть; GARCH для моделирования волатильности; модели машинного обучения типа Prophet, LSTM, особенно если добавлять внешние факторы).
Добавление экзогенных переменных (другие экономические показатели).
Более тщательный подбор гиперпараметров модели (например, с помощью GridSearch для ARIMA, если использовать pmdarima).
Анализ остатков модели для проверки ее адекватности.




### Вариант 2 для Евгения Володина
Практическая часть

 Ниже приведён пример, как можно организовать практическую часть дипломной работы (MVP для прогнозирования курса доллара США) и написать на Python программу для базового прогнозирования по имеющимся данным из вашего Excel-файла.Данная практическая часть может быть включена в дипломный проект в виде главы или приложения, где вы, опираясь на теоретические основы из основной части («Диплом.docx») и общие рекомендации по структуре работы («Гайд по дипломной работе Володина.pdf»), демонстрируете применение методов машинного (или статистического) обучения для решения конкретной задачи — прогнозирования курса валют.1. Структура практической частиНиже — ориентировочный план раздела «Практическая часть. MVP для прогнозирования курса валют»:
Постановка задачи

Кратко сформулировать задачу: «Необходимо спрогнозировать курс доллара США к рублю на основании исторических данных, скачанных с сайта ЦБ РФ».
Указать, что в качестве MVP создаётся упрощённая система прогнозирования без глубокой оптимизации.

#### Описание данных

#### Сказать, что исходная таблица (файл RC_F01_05_2020_T22_05_2025.xlsx) содержит столбцы:

nominal — номинал валюты (всегда 1 для USD);
data — числовое представление даты (в формате Excel или в виде порядкового числа);
curs — курс доллара США к рублю на соответствующую дату;
cdx — текстовое обозначение валюты, здесь «Доллар США».

Указать, что нужно преобразовать data в формат «год-месяц-день» или datetime для корректной работы с временными рядами.

#### Обработка и визуализация

Показать, как данные считываются, очищаются от «пропусков» и дубликатов, приводятся к нужному формату (датам).
Построить простой график «Дата – Курс» для первичного анализа динамики.

Выбор и обучение простой модели

Обосновать выбор типа модели (например, классический ARIMA / SARIMAX или модель на базе популярной библиотеки Prophet, или упрощённая линейная регрессия).
Критически отметить, что для полноценных производственных решений необходимы дополнительные факторы (нефтяные котировки, макроэкономические показатели и т.п.), но для MVP можно ограничиться историческим временным рядом.

Прогнозирование и оценка

Разбить исторические данные на «train» и «test», обучить модель на train.
Сделать прогноз на несколько дней/недель вперёд и сравнить с фактическими значениями (если есть).
Вычислить метрики (MAE, MSE, RMSE или MAPE) и сделать короткие выводы о том, насколько точно MVP предсказывает курс.

Выводы и дальнейшие шаги

Отметить, что в реальном проекте систему можно дополнительно улучшить:

Добавить внешние факторы (нефть, инфляция и т.д.).
Использовать другие модели (LSTM, Transformers, регрессия на множествах признаков и пр.).
Сделать автоматизированный веб-сервис или чат-бот для вывода прогноза.

- Пример Python-кода (MVP)Ниже представлен пример минимального кода на Python, демонстрирующего основные шаги: загрузка данных из Excel, подготовка дат, обучение простой ARIMA-модели (через библиотеку statsmodels) и краткий прогноз.
  Важно: Для запуска необходимы библиотеки pandas, matplotlib, statsmodels (устанавливаются через pip install pandas matplotlib statsmodels openpyxl).
  При использовании Jupyter Notebook не забудьте добавить магическую команду %matplotlib inline для корректного отображения графиков.

```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# 1. Загрузка данных из Excel
# Предположим, что файл "RC_F01_05_2020_T22_05_2025.xlsx" находится в текущей папке.
df = pd.read_excel("RC_F01_05_2020_T22_05_2025.xlsx", sheet_name="RC")

# 2. Предобработка данных
# В вашем файле столбец с датой называется "data" и хранит числовой формат Excel.
# Для преобразования числовых значений Excel-даты в формат datetime можно сделать:
#   - По умолчанию хронологией Excel является 1899-12-30, но проверяйте смещения.
#   - Ниже пример на основе частой ситуации, когда 2 января 1900 это "2" и т.д.

# Функция для преобразования даты из формата Excel:
def excel_date_to_datetime(excel_daynum):
    # Excel (1900-based date system) + timedelta
    return datetime(1899, 12, 30) + timedelta(days=excel_daynum)

df['date'] = df['data'].apply(excel_date_to_datetime)

# Убедимся, что столбец "date" стал датой. Удалим лишние колонки при необходимости.
df = df[['date','curs']].copy()

# Отсортируем по дате, т.к. в исходном файле порядок может быть разный
df = df.sort_values(by='date').reset_index(drop=True)

# 3. Визуальный анализ
plt.figure(figsize=(10,4))
plt.plot(df['date'], df['curs'], label="Курс USD/RUB")
plt.title("Исторический курс доллара к рублю (CBR)")
plt.xlabel("Дата")
plt.ylabel("Курс")
plt.legend()
plt.show()

# 4. Разделим данные на train/test для оценки
# Допустим, возьмём последние ~30 дней в качестве теста
train_size = len(df) - 30
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Строим модель ARIMA
# Для упрощения возьмём параметры (p,d,q) вручную (например (1,1,1)).
# В реальном проекте параметры лучше подобрать через auto_arima или другие методики.
train_ts = train_df.set_index('date')['curs']

model = ARIMA(train_ts, order=(1,1,1))
model_fit = model.fit()

# 5. Прогнозируем на длину test_df
forecast_steps = len(test_df)
forecast = model_fit.forecast(steps=forecast_steps)

# 6. Сравнение прогноза с реальными данными
test_df = test_df.copy()  # чтобы не было предупреждения SettingWithCopy
test_df['forecast'] = forecast.values

plt.figure(figsize=(10,4))
plt.plot(train_df['date'], train_df['curs'], label="Train")
plt.plot(test_df['date'], test_df['curs'], label="Test - Факт", color='green')
plt.plot(test_df['date'], test_df['forecast'], label="Test - Прогноз", color='red', linestyle='--')
plt.title("Прогноз курса доллара (ARIMA)")
plt.xlabel("Дата")
plt.ylabel("Курс USD/RUB")
plt.legend()
plt.show()

# 7. Оценка точности (например, RMSE)
rmse = np.sqrt(np.mean((test_df['curs'] - test_df['forecast'])**2))
print(f"RMSE прогноза: {rmse:.3f}")

# При желании можно посчитать MAPE (процентную ошибку):
mape = np.mean(np.abs((test_df['curs'] - test_df['forecast']) / test_df['curs'])) * 100
print(f"MAPE прогноза: {mape:.2f}%")
```

#### Краткие пояснения к коду

#### Чтение Excel
Используется pd.read_excel с указанием sheet_name="RC", так как во вложенном файле нужный лист называется «RC». При необходимости можно распечатать df.head(), чтобы убедиться, что данные считались корректно.

#### Преобразование Excel-даты
В данных data содержит порядковый номер дня (как это устроено в Excel). Поэтому мы пишем функцию excel_date_to_datetime, которую применяем к каждому числу, чтобы получить datetime.

#### Разделение на train/test
Нужно, чтобы проверить, насколько адекватно модель предсказывает будущие значения. Можно менять размер train, test по своему усмотрению (например, 70/30, 80/20 и т.д.).

#### Простая ARIMA (1,1,1)

Параметры (p, d, q) можно подобрать вручную, оценив автокорреляцию и частную автокорреляцию, либо воспользоваться pmdarima.arima.auto_arima, которая сама перебирает параметры.
В MVP достаточно продемонстрировать, как базовая модель строится и выдаёт прогноз.

#### Визуализация прогноза

На графике видно синюю линию (train), зелёную (фактический test) и красную (прогноз на test).
Так понятно, насколько прогноз укладывается в реальный «коридор» значений курса.

#### Метрики

RMSE показывает «среднеквадратичное отклонение»; чем меньше, тем лучше.
MAPE (Mean Absolute Percentage Error) показывает среднюю процентную ошибку.

### - 3 Как включить в диплом
  Ниже — пример того, как можно описать данную работу в тексте диплома (применительно к материалам, представленным в «Диплом.docx»). Разумеется, вы адаптируете стилистику и объём под свои требования:

#### Название раздела (пример):
"Глава 3. Практическая реализация системы ИИ-прогнозирования курса валют (MVP)"
3.1. Постановка задачи
В рамках данной работы была разработана минимально необходимая версия (MVP) системы прогнозирования курса доллара США на основе временного ряда, загруженного с официального сайта ЦБ РФ. Целью является проверить возможность применения методов машинного обучения (на примере модели ARIMA) для прогноза краткосрочной динамики USD/RUB.
3.2. Описание данных
Данные представлены в Excel-файле «RC_F01_05_2020_T22_05_2025.xlsx», содержащем столбцы nominal, data, curs и т. д. Основной интерес представляет временной ряд «curs», указывающий значение курса доллара на определённую дату, хранящуюся в числовом формате Excel.
[…]
3.3. Выбор и обучение модели
Для простоты и наглядности в MVP выбрана классическая модель ARIMA (Auto-Regressive Integrated Moving Average). Она позволяет обнаружить и использовать автокорреляционные свойства временного ряда. Параметры (p, d, q) подобраны эвристически (1,1,1). В качестве инструментария использована библиотека statsmodels на языке Python.
[…]
3.4. Результаты и оценка качества
Проведён эксперимент, в ходе которого последняя часть выборки (30 дней) использовалась как тест. Полученный RMSE составил X.XXX, а MAPE — Y.YY%. Данные метрики подтверждают, что в рамках MVP модель в целом способна улавливать тренд, однако прогнозы могут содержать отклонения, особенно в периоды резких колебаний курса.
В перспективе улучшения к точности могут быть достигнуты путём:

учёта сторонних признаков (мировые экономические новости, цена на нефть и т.д.);
подбора оптимальных гиперпараметров;
использования более продвинутых алгоритмов (например, LSTM).

3.5. Выводы
Разработанная MVP-система демонстрирует базовую возможность машинного обучения для краткосрочного прогноза курса валют. При дальнейшей доработке (учёт фундаментальных факторов, настройка модели, расширение тестируемых гипотез и т.д.) система может использоваться брокерами, частными инвесторами и иными заинтересованными лицами для оперативного анализа финансовых рисков.
[…]

Основные замечания

Данные. Убедитесь, что даты после преобразования из Excel-чисел корректны. Иногда бывают сдвиги (так называемая «ошибка 1900 года»). Проверьте, что даты совпадают с реальными (1 мая 2020, 2 мая 2020 и т.д.).
Предобработка. Желательно убрать дубликаты, пропуски (если есть), а также выбросы (аномальные значения).
Оптимизация параметров. Для настоящей рабочей системы стоит хотя бы протестировать разные (p, d, q) или использовать auto_arima. Но в MVP можно ограничиться одним вариантом.
Будущее расширение. Если хотите сделать прогноз более надёжным, добавьте дополнительные столбцы (например, цену на нефть) и обучите регрессию или многомерную временную модель (SARIMAX).



Короче я сделал по  рекомендациям из твоего Гайда
