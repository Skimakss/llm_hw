# hw2_parallel_pretrain

## Цель работы

В этом ДЗ исследуется обучение causal language model на 2 GPU с использованием двух подходов к распределённому обучению:

- **DeepSpeed**
- **FSDP** (`Fully Sharded Data Parallel`)

Цели эксперимента:

- сравнить DeepSpeed и FSDP на одинаковом пайплайне;
- измерить скорость обучения, потребление памяти и итоговый `eval_loss`;
- оценить, насколько близко масштабирование к линейному при переходе от 1 GPU к 2 GPU;
- проанализировать trade-off между скоростью, памятью и качеством.

---

## Базовый пайплайн из hw1

В работе используется тот же пайплайн, что и в `hw1`:

- **датасет**: `wikimedia/wikipedia`, subset `20231101.ru`
- **токенизатор**: `ai-forever/rugpt3small_based_on_gpt2`
- **max length**: `512`
- **задача**: causal LM
- `labels` создаются как копия `input_ids`
- токенизированный датасет заранее сохранён в `parquet`-шарды
- используется фиксированный split без shuffle:
  - первые `5000` объектов — в `eval`
  - остальные — в `train`

---

## Архитектура модели

Использовалась модель `Qwen3ForCausalLM` с конфигом порядка **1B параметров**.

Число параметров:

\[
N = 960{,}881{,}664
\]

---

## 1. Сетап экспериментов

## 1.1. Smoke/debug этап на Kaggle

Kaggle использовался как этап отладки и предварительного сравнения режимов.

### Цели smoke-этапа

- проверить запуск обучения на 2 GPU через `accelerate`;
- отладить DeepSpeed и FSDP;
- получить первые измерения памяти и скорости;
- выбрать кандидатов для финальных 30-минутных запусков.

### Smoke setup

- GPU: **2×Tesla T4**
- mixed precision: `bf16`
- короткие smoke-run'ы:
  - `train = 128`
  - `eval = 16`

---

## 1.2. Финальный setup на Runpod

Финальные 30-минутные эксперименты выполнялись на **2×A100 80GB**.

### Окружение

- GPU: **2×NVIDIA A100-SXM4-80GB**
- Python: **3.11.10**
- CUDA runtime на хосте: **12.7**
- базовый образ pod:
  - `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### Версии библиотек

- `torch == 2.4.1+cu124`
- `transformers == 5.0.0`
- `datasets == 4.8.3`
- `accelerate == 1.12.0`
- `deepspeed == 0.18.9`
- `flash-attn == 2.7.3`

---

## 2. Конфиги и команды запуска

## 2.1. Запуск на 2 GPU

Базовый способ запуска на 2 GPU:

```bash
accelerate launch --num_processes 2 --main_process_port 0 parallel_train.py
```

---

## 2.2. Конфиг финального эксперимента

Для финальных запусков использовался конфиг **candidate-4**.

### Почему был выбран candidate-4

Для `hw2` в качестве reference-конфига был выбран **candidate-4**, потому что на запуске на **1×A100** он показал **более стабильный loss**, чем более агрессивный `candidate-5`.

Используемый финальный конфиг:

```python
A100_CANDIDATE_4 = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 20,
    "learning_rate": 4e-4,
    "lr_scheduler_type": "linear",
    "warmup_steps": 0,
    "torch_compile": False,
    "optim": "adamw_torch",
}
```

Глобальный batch size:

\[
8 \cdot 20 \cdot 2 = 320 objects
\]

---

## 2.3. Финальные режимы

### DeepSpeed

Для DeepSpeed в финальном запуске использовался:

- `ZeRO-3`

### FSDP

Для FSDP в финальном запуске использовался:

- `full_shard auto_wrap`

---

## 3. Теоретическая оценка памяти

Ниже приведена грубая оценка **статической памяти на одну GPU**, где учитываются только:

- параметры модели
- градиенты
- состояния оптимизатора

То есть это оценка **train states**, без учёта:

- активаций
- временных CUDA-буферов
- communication overhead
- reserved memory / fragmentation

---

## 3.1. Baseline

Для `AdamW`:

- веса: 4 байта на параметр
- градиенты: 4 байта на параметр
- состояния `AdamW` (`m`, `v`): 8 байт на параметр

Итого:

\[
4 + 4 + 8 = 16 \text{ байт на параметр}
\]

Для модели размера \(N \approx 10^9\):

\[
16N \approx 16 \cdot 10^9 \text{ bytes} \approx 16 \text{ GB}
\]

Вывод: при лимите 16 GB VRAM baseline/DDP с `AdamW` практически не оставляет памяти на активации и не помещается.

---

## 3.2. ZeRO-1 + AdamW

Для `ZeRO-1` на 2 GPU:

- веса: \(4N\)
- градиенты: \(4N\)
- состояния оптимизатора: \(\frac{8N}{2}\)

Итого:

\[
4N + 4N + \frac{8N}{2} = 12N
\]

Подстановка:

\[
12 \cdot 960{,}881{,}664 = 11{,}530{,}579{,}968 \text{ bytes} \approx 11.5 \text{ GB}
\]

---

## 3.3. ZeRO-2 + AdamW

Для `ZeRO-2` на 2 GPU:

- веса: \(4N\)
- градиенты: \(\frac{4N}{2}\)
- состояния оптимизатора: \(\frac{8N}{2}\)

Итого:

\[
4N + \frac{4N}{2} + \frac{8N}{2} = 10N
\]

Подстановка:

\[
10 \cdot 960{,}881{,}664 = 9{,}608{,}816{,}640 \text{ bytes} \approx 9.6 \text{ GB}
\]

---

## 3.4. ZeRO-3 + AdamW

Для `ZeRO-3` на 2 GPU шардируются:

- параметры
- градиенты
- состояния оптимизатора

Оценка:

- параметры: \(\frac{4N}{2}\)
- градиенты: \(\frac{4N}{2}\)
- состояния оптимизатора: \(\frac{8N}{2}\)

Итого:

\[
\frac{4N + 4N + 8N}{2} = 8N
\]

Подстановка:

\[
8 \cdot 960{,}881{,}664 = 7{,}687{,}053{,}312 \text{ bytes} \approx 7.7 \text{ GB}
\]

---

## 3.5. Сводная таблица теоретической оценки

| Mode | Теоретическая статическая память на одну GPU |
|---|---:|
| baseline / DDP | ~15.4 GB |
| ZeRO-1 | ~11.5 GB |
| ZeRO-2 | ~9.6 GB |
| ZeRO-3 | ~7.7 GB |

---

## 4. Сравнение теории и реальных измерений на Kaggle

По итогам smoke-run'ов были измерены:

- `peak_allocated_mb`
- `peak_reserved_mb`

### Таблица сравнения

| Mode | Теория, GB | Peak allocated, GB | Peak reserved, GB |
|---|---:|---:|---:|
| ZeRO-1 | 11.5 | 11.0 | 13.1 |
| ZeRO-2 | 9.6 | 11.0 | 12.9 |
| ZeRO-3 | 7.7 | 11.5 | 14.5 |

### Анализ

Из таблицы видно, что:

- для **ZeRO-1** грубая теоретическая оценка хорошо совпала с реальным `peak_allocated`;
- для **ZeRO-2** и особенно **ZeRO-3** реальные peak values оказались выше теоретической оценки статической памяти.

Это ожидаемо, так как теоретическая оценка не учитывала:

- активации
- временные CUDA-буферы
- communication buffers
- накладные расходы DeepSpeed
- работу allocator
- reserved memory / fragmentation

### Вывод

Грубая теоретическая оценка хорошо объясняет снижение memory pressure при переходе от baseline к ZeRO-режимам, однако для практического сравнения режимов необходимо ориентироваться на **реальные измерения peak memory**.

---

## 5. Smoke-результаты на Kaggle

Для smoke-тестов использовался укороченный конфиг:

- `RUN_MODE = "smoke"`
- `SMOKE_TRAIN_SIZE = 128`
- `SMOKE_EVAL_SIZE = 16`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 1`
- `save_steps = 100`
- `save_strategy = "no"`
- `eval_strategy = "no"`
- `load_best_model_at_end = False`
- `logging_steps = 10`
- `warmup_steps = 0`
- `lr_scheduler_type = "constant"`
- `torch_compile = False`
- `ddp_find_unused_parameters = False`

Для сопоставимого smoke-сетапа (`train=128`, `eval=16`, `2×T4`, `bf16`) были измерены скорость обучения и peak memory.

| Mode | train_runtime (s) | train_steps_per_second | train_loss | peak_allocated_mb | peak_reserved_mb |
|---|---:|---:|---:|---:|---:|
| ZeRO-1 | 143.60 | 0.446 | 9.291 | 11012.53 | 13118.0 |
| ZeRO-2 | 145.31 | 0.440 | 9.405 | 11012.53 | 12928.0 |
| ZeRO-3 | 125.51 | 0.510 | 9.287 | 11505.17 | 14530.0 |
| FSDP | **113.82** | **0.562** | 9.360 | **8132.53** | **10986.0** |

### Выводы по smoke-этапу

- На smoke-run **FSDP** показал лучший throughput и минимальную память.
- Среди режимов DeepSpeed наиболее удачным кандидатом для финального прогона выглядел **ZeRO-3**.
- Поэтому для финальных запусков были выбраны:
  - **DeepSpeed ZeRO-3**
  - **FSDP full_shard auto_wrap**


---

## 6. Финальные 30-минутные прогоны на 2×A100

### Финальный сетап на Runpod

Финальные 30-минутные эксперименты проводились на **Runpod** в окружении с двумя GPU:

- **GPU:** `2× NVIDIA A100-SXM4-80GB`
- **CUDA (по `nvidia-smi`):** `12.7`
- **Python:** `3.11.10`
- **PyTorch:** `2.4.1+cu124`
- **Transformers:** `5.0.0`
- **Datasets:** `4.8.3`
- **Accelerate:** `1.12.0`
- **DeepSpeed:** `0.18.9`
- **Flash Attention:** `2.7.3`

### Временный patch для совместимости FSDP

При запуске FSDP в окружении Runpod со стеком `torch 2.4.1 + transformers 5.0.0` возникала ошибка, связанная с отсутствием в текущей версии PyTorch атрибута `torch.distributed.fsdp.register_fsdp_forward_method`, который ожидался внутри `transformers` в FSDP-пути `Trainer`.

Чтобы обойти эту несовместимость и запустить эксперимент, был добавлен небольшой временный compatibility patch:

```python
import torch.distributed.fsdp as torch_fsdp

if not hasattr(torch_fsdp, "register_fsdp_forward_method"):
    torch_fsdp.register_fsdp_forward_method = lambda *args, **kwargs: None
```

## 6.1. DeepSpeed ZeRO-3

Финальные метрики:

- `train_runtime = 1801.605 s`
- `train_steps_per_second = 3.365`
- `train_loss = 7.0235`
- `peak_allocated_mb = 18829.27`
- `peak_reserved_mb = 26434.0`
- `seen_samples = 125440`
- `eval_loss = 5.22377`

Дополнительно:

- `eval_runtime = 20.2689 s`
- `eval_steps_per_second = 7.746`

---

## 6.2. FSDP

Финальные метрики:

- `train_runtime = 1802.138 s`
- `train_steps_per_second = 3.364`
- `train_loss = 6.9759`
- `peak_allocated_mb = 16411.90`
- `peak_reserved_mb = 19902.0`
- `seen_samples = 130240`
- `eval_loss = 5.25966`

Дополнительно:

- `eval_runtime = 21.1006 s`
- `eval_steps_per_second = 7.441`

---

## 6.3. Итоговая таблица финальных запусков

| Mode | train_runtime, s | train_steps_per_second | seen_samples (`k`) | peak_allocated_mb | peak_reserved_mb | eval_loss |
|---|---:|---:|---:|---:|---:|---:|
| FSDP | 1802.138 | 3.364 | 130240 | 16411.90 | 19902.0 | 5.25966 |
| DeepSpeed ZeRO-3 | 1801.605 | 3.365 | 125440 | 18829.27 | 26434.0 | **5.22377** |

---

## 7. Сравнение с 1 GPU

В качестве reference используется запуск на **1×A100** для `candidate-4`.

\[
k_{1gpu} = 73760
\]

Для финальных 2-GPU запусков:

\[
k_{2gpu}^{FSDP} = 130240
\]

\[
k_{2gpu}^{DS} = 125440
\]

Отношение ускорения:

\[
r_{FSDP} = \frac{130240}{73760} \approx 1.77
\]

\[
r_{DS} = \frac{125440}{73760} \approx 1.70
\]

---

## 8. Анализ scaling

Идеально линейное масштабирование при переходе от 1 GPU к 2 GPU соответствовало бы:

\[
r = 2
\]

В обоих режимах получилось:

- `r_FSDP ≈ 1.77`
- `r_DS ≈ 1.70`

То есть в обоих случаях:

\[
r < 2
\]

### Возможные причины, почему scaling не линейный

- communication overhead между GPU;
- синхронизация градиентов / шардов;
- overhead distributed runtime;
- дополнительные буферы и операции шардирования;
- input pipeline и другие non-compute overhead;
- накладные расходы allocator и fragmentation.

Такой результат выглядит ожидаемо для реального distributed training.

---

## 9. Анализ устойчивости и качества

По итогам финальных прогонов:

- оба режима показали сопоставимую динамику loss;
- оба режима успешно прошли финальный `eval`;
- значения `eval_loss` близки.

Сравнение:

- **DeepSpeed ZeRO-3** дал немного лучший `eval_loss`
- **FSDP** обработал больше семплов за тот же бюджет времени и потребовал заметно меньше памяти

---

## 10. Итоговый вывод

### По скорости

FSDP и DeepSpeed ZeRO-3 показали практически одинаковую скорость:

- `3.364 steps/s` для FSDP
- `3.365 steps/s` для DeepSpeed

### По памяти

FSDP оказался заметно экономнее:

- `peak_allocated_mb`: **16411.90** против **18829.27**
- `peak_reserved_mb`: **19902.0** против **26434.0**

### По количеству обработанных семплов за 30 минут

- **FSDP:** `130240`
- **DeepSpeed ZeRO-3:** `125440`

То есть FSDP дал немного больший throughput по фактически обработанным семплам.

### По качеству

- **DeepSpeed ZeRO-3:** `eval_loss = 5.22377`
- **FSDP:** `eval_loss = 5.25966`

DeepSpeed дал немного лучший итоговый `eval_loss`, однако отличие небольшое, и вполне возможно, что при статистическом тесте оно может оказаться не значимым.

### Общий вывод

Для данной модели и данного 30-минутного бюджета, в целом, **FSDP выглядит более удачным компромиссом** для данной задачи, поскольку даёт:

- сопоставимую скорость,
- меньшее потребление памяти,
- немного больше обработанных семплов за тот же бюджет времени.

---

## 11. Структура проекта

В папке `hw2_parallel_pretrain` используются следующие основные файлы:

- `parallel_train.py` — основной скрипт запуска обучения; поддерживает режимы DeepSpeed и FSDP, запускает train/eval и сохраняет итоговые метрики.
- `config.py` — центральный конфиг эксперимента: `RUN_MODE`, `PARALLEL_MODE`, candidate-конфиги, batch size, scheduler, precision и прочие параметры запуска.
- `data_utils.py` — загрузка токенизированного датасета из parquet-шардов, подготовка токенизатора и фиксированный train/eval split.
- `model_utils.py` — создание модели `Qwen3ForCausalLM`, настройка dtype и backend.
- `runtime_utils.py` — вспомогательная логика runtime: валидация окружения, W&B, расчёт метрик (`seen_samples`, memory metrics), сохранение конфигов и служебные функции для финального eval.
- `callbacks.py` — callback для ограничения обучения по времени (`30 минут`).
- `ds_zero1.json` — конфиг DeepSpeed для ZeRO Stage 1.
- `ds_zero2.json` — конфиг DeepSpeed для ZeRO Stage 2.
- `ds_zero3.json` — конфиг DeepSpeed для ZeRO Stage 3.
- `requirements.txt` — зависимости проекта.
- `wandb_assets/` — графики для отчёта из W&B.


## 12. Структура проекта

В папке `llm_parallel_pretrain` используются следующие основные файлы:

- `parallel_train.py` — основной скрипт запуска обучения; поддерживает режимы DeepSpeed и FSDP, запускает train/eval и сохраняет итоговые метрики.
- `config.py` — центральный конфиг эксперимента: `RUN_MODE`, `PARALLEL_MODE`, candidate-конфиги, batch size, scheduler, precision и прочие параметры запуска.
- `data_utils.py` — загрузка токенизированного датасета из parquet-шардов, подготовка токенизатора и фиксированный train/eval split.
- `model_utils.py` — создание модели `Qwen3ForCausalLM`, настройка dtype и backend.
- `runtime_utils.py` — вспомогательная логика runtime: валидация окружения, W&B, расчёт метрик (`seen_samples`, memory metrics), сохранение конфигов и служебные функции для финального eval.
- `callbacks.py` — callback для ограничения обучения по времени (`30 минут`).
- `ds_zero1.json` — конфиг DeepSpeed для ZeRO Stage 1.
- `ds_zero2.json` — конфиг DeepSpeed для ZeRO Stage 2.
- `ds_zero3.json` — конфиг DeepSpeed для ZeRO Stage 3.
- `requirements.txt` — зависимости проекта.
- `wandb_assets/` — графики для отчёта из W&B.

---

## Краткая инструкция по запуску режимов

Перед запуском нужно убедиться, что:

- в `config.py` выставлен нужный `RUN_MODE` (`smoke` или `final`);
- выбран нужный `PARALLEL_MODE`;
- для DeepSpeed указан нужный `DEEPSPEED_CONFIG_PATH`;
- для FSDP указан `FSDP_MODE`;

### Использование проекта с двумя вариантами загрузки данных

В проекте предусмотрены два сценария работы с данными:

1. использовать **уже готовый токенизированный датасет** в виде parquet-шардов;
2. автоматически **скачать raw dataset, токенизировать его и сохранить parquet-шарды** перед обучением.

Выбор сценария выполняется через флаг в `config.py`.

---

### 12.1 Переключатель в `config.py`

В конфиге используется флаг:

```python
USE_PRETOKENIZED_DATASET = False
```

Возможны два режима:

- `USE_PRETOKENIZED_DATASET = True`  
  код ожидает, что parquet-шарды уже существуют и просто читает их через `load_tokenized_dataset()`;

- `USE_PRETOKENIZED_DATASET = False`  
  код сначала вызывает `prepare_dataset(...)`, сохраняет parquet-шарды в `DATA_DIR`, а затем загружает их через `load_tokenized_dataset()`.

---

### 12.2 Вариант 1 — использовать готовые parquet-шарды

Этот режим удобен, если токенизация уже была выполнена заранее и вы не хотите повторять её при каждом запуске.

### Что нужно подготовить

В каталоге, указанном в `config.py` как `DATA_DIR`, должна лежать папка с parquet-файлами, например:

```text
tokenized_data/
├── 00000.parquet
├── 00001.parquet
├── ...
└── 00031.parquet
```

По умолчанию путь задаётся так:

```python
DATA_DIR = os.path.join(os.path.dirname(__file__), "tokenized_data")
```

То есть рядом с кодом должна быть папка `tokenized_data/`.

### Что выставить в `config.py`

```python
USE_PRETOKENIZED_DATASET = True
```

### 12.3 Вариант 2 — подготовить датасет автоматически с нуля

Этот режим удобен, если parquet-шарды ещё не были созданы.

### Что выставить в `config.py`

```python
USE_PRETOKENIZED_DATASET = False
```
Во всех случаях запуск на 2 GPU выполняется через:

```bash
export WANDB_API_KEY="..."
accelerate launch --num_processes 2 --main_process_port 0 parallel_train.py
```

