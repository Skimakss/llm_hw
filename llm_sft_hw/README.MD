# SFT Homework

## Что это

В проекте выполнен рефакторинг ноутбука по домашнему заданию по **Supervised Fine-Tuning (SFT)** и собран удобный пайплайн для запуска экспериментов в Kaggle.

Вместо большого ноутбука с множеством ячеек логика была вынесена в отдельные модули, чтобы:
- было проще переиспользовать код;
- удобно запускать новые эксперименты;
- отдельно управлять данными, токенизатором, моделью, обучением и метриками;
- логировать результаты в W&B.

---

## Что требовалось реализовать

В домашнем задании нужно было:
- подготовить SFT-пайплайн на датасете `databricks/databricks-dolly-15k`;
- обучить модель `Qwen/Qwen2.5-0.5B-Instruct`;
- сравнить режимы **baseline** и **packing**;
- посчитать метрики:
  - скорость обучения;
  - `loss`;
  - `perplexity`;
  - `JSON validity`;
  - `Function Calling EM@20`;
- показать примеры генераций до и после обучения;
- оформить результаты экспериментов.

---

## Что реализовано

### 1. Рефакторинг ноутбука в модули

Код из исходного ноутбука разложен по модулям:

- `sft/config.py` — dataclass-конфиги;
- `sft/data.py` — подготовка данных, аугментации, tool-use, split;
- `sft/pipeline.py` — сборка data/tokenizer pipeline;
- `sft/tokenizer_utils.py` — работа с токенизатором, tool tokens, padding vocab;
- `sft/model_utils.py` — загрузка модели и синхронизация эмбеддингов;
- `sft/training.py` — сборка trainer и запуск обучения;
- `sft/metrics.py` — генерация, JSON validity, Function Calling EM;
- `run_sft.py` — базовая точка входа.

### 2. Подготовка данных

Реализован пайплайн:
- загрузка `databricks/databricks-dolly-15k`;
- приведение разных форматов примеров к единому виду `messages`;
- добавление `system`-сообщения;
- генерация части multi-turn диалогов;
- искусственная аугментация примеров с tool-use;
- разбиение на `train/validation`.

### 3. Подготовка токенизатора

Реализовано:
- добавление недостающих tool tokens;
- выравнивание размера словаря до числа, кратного 128;
- ресайз эмбеддингов модели под обновлённый tokenizer.

### 4. Обучение

Были проведены эксперименты в режимах:
- baseline (`packing=False`)
- packing (`packing=True`)

Обучение запускалось в Kaggle на GPU. По ходу работы было установлено, что для данного сетапа режим `fp16` приводил к дивергенции лосса, а стабильное обучение достигалось в **bf16** и **fp32**.

### 5. Метрики

Реализован расчёт:
- `train_runtime`
- `train_samples_per_second`
- `train_steps_per_second`
- `final_train_loss`
- `final_eval_loss`
- `perplexity`
- `JSON validity@100`
- `Function Calling EM@20`

### 6. Логирование экспериментов

Добавлено логирование в **Weights & Biases (W&B)**:
- конфигов запусков;
- train/eval метрик;
- итоговых метрик;
- результатов сетки экспериментов.

---

## Кратко про проведённые эксперименты

### Baseline vs Packing

Сравнение baseline-запусков:

- `baseline_bf16_packing_false`
  - `final_eval_loss = 1.6987`
  - `perplexity = 5.4667`
  - `train_runtime = 392.9 s`

- `baseline_bf16_packing_true`
  - `final_eval_loss = 3.9985`
  - `perplexity = 54.5167`
  - `train_runtime = 244.4 s`

**Вывод:** `packing=True` ускорил обучение, но сильно ухудшил качество на этом датасете и в этом сетапе.

### Мини-сетка экспериментов

Была проведена сетка из 6 запусков по `learning_rate` и `batch_size`.

Лучший запуск:

- `grid6_run3_ep1_lr2e-05_bs1_bf16_packFalse`
  - `learning_rate = 2e-5`
  - `batch_size = 1`
  - `grad_accum_steps = 4`
  - `packing = False`
  - `torch_dtype = bf16`
  - `final_eval_loss = 1.6236`
  - `perplexity = 5.0715`
  - `JSON validity@100 = 1.0`
  - `Function Calling EM@20 = 1.0`

### Основные выводы

- Лучшее качество в рамках рассмотренной сетки дало сочетание:
  - `learning_rate = 2e-5`
  - `batch_size = 1`
  - `packing = False`
  - `bf16`
- При увеличении `batch_size` обучение становилось быстрее, но качество ухудшалось.
- Метрики `JSON validity` и `EM@20` получились очень высокими, но их нужно интерпретировать осторожно, так как тестовые промпты были достаточно шаблонными.

---

## Структура файлов в папке

Сейчас в проектной папке находятся:

- `SFT_hw_report.pdf` — оформленный отчёт по домашней работе;
- `sft_kaggle_notebook.ipynb` — основной Kaggle-ноутбук с экспериментами;
- `sft_refactor.zip` — архив с модульной версией пайплайна;
- `wandb_experiment_configs.csv` — выгрузка конфигов экспериментов из W&B.

При необходимости можно также выгружать отдельный CSV с итоговыми метриками и history run-ов из W&B.

---

## Как запускать на Kaggle

### Вариант, который использовался в работе

На Kaggle архив `sft_refactor.zip` загружался как **Dataset** и подключался к ноутбуку через **Add Input**.

Дальше использовалась такая схема:

1. Создать новый Kaggle Notebook.
2. Подключить dataset с `sft_refactor.zip`.
3. Распаковать или использовать уже распакованную папку `sft_refactor` из input.
4. Добавить путь проекта в `sys.path`.
5. Импортировать модули пайплайна.
6. Подготовить:
   - `data_cfg`
   - `model_cfg`
   - `train_cfg`
7. Запустить:
   - подготовку данных;
   - подготовку токенизатора;
   - обучение;
   - расчёт метрик;
   - логирование в W&B.

### Минимальная схема запуска в ноутбуке

```python
import sys

PROJECT_DIR = "/kaggle/input/datasets/skimaks/project/sft_refactor"
sys.path.insert(0, PROJECT_DIR)

from sft.config import DataConfig, ModelConfig, TrainConfig
from sft.pipeline import prepare_data_pipeline, prepare_tokenizer_pipeline, render_dataset_with_chat_template
```

Далее:

```python
data_cfg = DataConfig()
model_cfg = ModelConfig()
train_cfg = TrainConfig()

prepared_data = prepare_data_pipeline(data_cfg)
prepared_tokenizer = prepare_tokenizer_pipeline(model_cfg)
dataset_text = render_dataset_with_chat_template(
    prepared_data.dataset_messages,
    prepared_tokenizer.tokenizer,
)
```

После этого можно собирать модель, trainer и запускать эксперименты.

---

## Что стоит помнить при запуске

- Для Kaggle output-директорию лучше указывать в `/kaggle/working/...`, а не в `/kaggle/input/...`, потому что input read-only.
- Для новых сравнимых запусков модель нужно загружать **fresh**, а не переиспользовать уже обученный объект.
- После серии запусков полезно удалять чекпоинты и чистить память GPU, чтобы не переполнить `/kaggle/working`.
- Для логирования экспериментов удобно использовать отдельный W&B run на каждый запуск.

---

## Итог

В результате получился воспроизводимый и удобный SFT-пайплайн, который:
- работает в Kaggle;
- поддерживает рефакторинг исходного ноутбука в модули;
- позволяет быстро запускать новые эксперименты;
- считает нужные метрики;
- логирует результаты в W&B;
- подходит для оформления результатов домашней работы.
