# RAG-система: LangChain + Qdrant + Qwen3.5-4B

Проект реализует RAG-пайплайн на LangChain с Qdrant в роли векторной БД и локальным инференсом Qwen3.5-4B. Система поддерживает русский и английский языки и может работать в двух режимах:

- на удаленном сервере как Docker-сервис с HTTP API;
- локально через Python CLI и тот же Qdrant в Docker.

## Архитектура

| Слой | Технология | По умолчанию |
|---|---|---|
| API | HTTP server на стандартной библиотеке Python | порт `8000` |
| RAG-пайплайн | LangChain LCEL | prompt -> retriever -> generator |
| Векторная БД | Qdrant | Docker-сервис `qdrant` |
| Эмбеддинги | `Octen/Octen-Embedding-0.6B` | 1024 измерений, 100+ языков |
| Генерация | `Qwen/Qwen3.5-4B` | актуальная Qwen3.5-модель на 4B параметров |
| Документы | LangChain loaders | `.txt`, `.md`, `.rst`, `.pdf` |

## Запуск на удаленном сервере в Docker

1. Установите Docker и Docker Compose plugin на сервере.

2. Создайте `.env`:

```bash
cp .env.example .env
```

3. Проверьте важные переменные в `.env`:

```env
RAG_API_BIND=0.0.0.0
RAG_API_PORT=8000
QDRANT_HOST=localhost
QDRANT_HTTP_BIND=127.0.0.1
QDRANT_GRPC_BIND=127.0.0.1
GENERATOR_MODEL=Qwen/Qwen3.5-4B
SYSTEM_PROMPT="Ты полезный RAG-ассистент. Отвечай только по контексту и на языке вопроса."
TORCH_DTYPE=float16
```

Внутри Docker Compose приложение само переопределяет `QDRANT_HOST` на `qdrant`, поэтому локальное значение `localhost` остается удобным для запуска без контейнера приложения. Порты Qdrant по умолчанию открыты только на хосте Docker, а наружу публикуется только API.

4. Соберите и запустите сервисы:

```bash
docker compose build
docker compose up -d
```

5. Индексируйте документы из папки `data/documents`:

```bash
curl -X POST http://SERVER_IP:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"data/documents","reset":true}'
```

При смене `EMBEDDING_MODEL` или `EMBEDDING_DIM` коллекцию Qdrant нужно пересоздать. Для этого используйте `reset=true`, как в примере выше.

6. Задайте вопрос:

```bash
curl -X POST http://SERVER_IP:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Что такое RAG?"}'
```

Проверка доступности:

```bash
curl http://SERVER_IP:8000/health
curl http://SERVER_IP:8000/stats
```

Если API открывается через `localhost`, но не открывается через реальный IP сервера, проверьте публикацию порта и сетевые правила:

```bash
grep '^RAG_API_BIND=' .env
docker compose ps
docker compose port rag 8000
ss -ltnp | grep ':8000'
```

Для внешнего доступа в `.env` должно быть `RAG_API_BIND=0.0.0.0`, а в `docker compose ps` порт должен выглядеть примерно как `0.0.0.0:8000->8000/tcp`, а не `127.0.0.1:8000->8000/tcp`. После изменения `.env` пересоздайте сервис:

```bash
docker compose up -d --force-recreate rag
```

Если порт опубликован на `0.0.0.0`, но внешний IP все равно не отвечает, откройте порт `8000/tcp` в firewall сервера и в правилах облачного провайдера/security group.


## CLI внутри Docker

Интерактивный режим работает через compose profile `cli`:

```bash
docker compose run --rm rag-cli
```

Запускает интерактивный CLI-чат.
В интерактивном режиме доступна команда `/clear`, которая очищает контекст текущего диалога.

```bash
docker compose run --rm rag-cli python main.py --query "What is Qdrant?"
```

Выполняет один вопрос и сразу завершает контейнер.

```bash
docker compose run --rm rag-cli python ingest.py --reset data/documents
```

Пересоздает коллекцию Qdrant и заново индексирует документы из `data/documents`.

Можно также переопределять команду основного сервиса:

```bash
docker compose run --rm rag python main.py --query "Что хранит Qdrant?"
```

## Обновление Docker после изменений

Если изменился только `.env`, пересобирать образ не нужно. Достаточно пересоздать сервис, чтобы Docker Compose заново прочитал переменные окружения:

```bash
docker compose up -d --force-recreate rag
```

Для CLI новый `.env` подхватывается при следующем запуске:

```bash
docker compose --profile cli run --rm rag-cli
```

Если менялись код, `Dockerfile` или `requirements.txt`, пересоберите образ без кеша:

```bash
docker compose build --no-cache rag-cli rag
```

Если менялись `EMBEDDING_MODEL` или `EMBEDDING_DIM`, после пересборки нужно пересоздать коллекцию и переиндексировать документы:

```bash
docker compose --profile cli run --rm rag-cli python ingest.py --reset data/documents
```

## Локальный запуск без контейнера приложения

Нужен Python 3.12. Qdrant все равно удобно запускать в Docker.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
docker compose up -d qdrant
python ingest.py --reset data/documents
python main.py --query "Что такое RAG?"
```

Важно запускать тесты и приложение из `.venv`. Системный Python 3.9 на Windows может падать на бинарных зависимостях LangChain.

## API

`GET /health`

Возвращает статус сервиса без загрузки LLM.

`GET /stats`

Возвращает имя коллекции, количество чанков и используемые модели.

`POST /ingest`

Индексирует файл или папку внутри `data`.

```json
{
  "path": "data/documents",
  "reset": true
}
```

`POST /query`

Выполняет RAG-запрос.

```json
{
  "question": "What is RAG?",
  "session_id": "user-1"
}
```

Ответ:

```json
{
  "question": "What is RAG?",
  "answer": "...",
  "sources": ["sample.txt"],
  "session_id": "user-1"
}
```

`session_id` необязателен. Если передавать один и тот же `session_id` в нескольких запросах, RAG использует короткую историю диалога для уточняющих вопросов вроде “А какой еще есть метод?”. Размер окна задается переменной `CONTEXT_WINDOW_TURNS`.

## RAGAS-оценка

В проекте реализован отдельный evaluation-пайплайн на RAGAS. Он берет вопросы из `eval/testset.jsonl`, прогоняет их через текущий RAG, передает в RAGAS поля `user_input`, `response`, `retrieved_contexts`, `reference` и сохраняет отчеты в `eval/results`.

Конфигурация находится в `eval/ragas_config.json`:

```json
{
  "metrics": [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall"
  ],
  "rag": {
    "lazy_generator": true,
    "top_k": 5,
    "index_path": "data/documents",
    "reset_index": false
  },
  "experiments": {
    "top_k": [3, 5],
    "chunk_size": [384, 512],
    "chunk_overlap": [48, 64],
    "embedding_model": ["Octen/Octen-Embedding-0.6B"],
    "embedding_dim": [1024],
    "generator_model": ["Qwen/Qwen3.5-4B"],
    "max_new_tokens": [384],
    "temperature": [0.0]
  }
}
```

Запуск локально:

```bash
python evaluate_ragas.py
python evaluate_ragas.py --config eval/ragas_config.json
python evaluate_ragas.py --metrics faithfulness answer_relevancy --top-k 3
python evaluate_ragas.py --index-path data/documents --reset-index
python evaluate_ragas.py --experiments
```

Запуск внутри CLI Docker:

```bash
docker compose build rag-eval
docker compose --profile eval run --rm rag-eval # Запуск оценки RAGAS
docker compose --profile eval run --rm rag-eval python evaluate_ragas.py --index-path data/documents --reset-index
docker compose --profile eval run --rm rag-eval python evaluate_ragas.py --experiments # Запуск экспериментов
```

После изменений в Python-коде или зависимостях сначала пересоберите `rag-eval`, иначе Docker запустит старую версию evaluator из уже собранного образа.

Обычная команда `docker compose --profile eval run --rm rag-eval` сама поднимает зависимости `qdrant` и `vllm`, а затем запускает оценку. Готовность OpenAI-compatible endpoint проверяется внутри `evaluate_ragas.py` перед расчетом метрик: скрипт ждет `/v1/models` до `RAGAS_LLM_WAIT_TIMEOUT` секунд.

Если `vllm` уже запускался раньше и остался в ошибочном состоянии после смены настроек, пересоздайте его отдельно и посмотрите логи:

```bash
docker compose --profile eval up -d --force-recreate vllm
docker compose logs -f vllm
```

RAGAS использует vLLM как OpenAI-compatible judge LLM. Внутри Docker Compose evaluator подключается к `http://vllm:8000/v1`; при локальном запуске Python с хоста по умолчанию используется `http://localhost:8001/v1`.
Рабочий OpenAI-compatible endpoint для RAGAS проверяется через `/v1/models`.

```bash
curl -H "Authorization: Bearer local-vllm-key" http://localhost:8001/v1/models
docker compose logs -f vllm
```

Для локального vLLM оценка RAGAS запускается с консервативным параллелизмом: `RAGAS_MAX_WORKERS=1`, `RAGAS_TIMEOUT=900` и `RAGAS_LLM_MAX_TOKENS=512`. Сам vLLM также ограничен по KV-cache: `VLLM_GPU_MEMORY_UTILIZATION=0.55`, `VLLM_MAX_MODEL_LEN=4096`, `VLLM_MAX_NUM_SEQS=1`. Это снижает риск `TimeoutError` и OOM на LLM-based метриках, особенно когда judge LLM, RAG-генератор и embeddings делят одну GPU.

Для подбора оптимальной комбинации для продакшена стоит использовать `experiments` в `eval/ragas_config.json`. 
Используется Декартово произведение: например `top_k=[3,5]`, `chunk_size=[384,512]`, `chunk_overlap=[48,64]` даст 8 запусков. 
Важно: Если меняете `chunk_size`, `chunk_overlap`, `embedding_model` или `embedding_dim`, включите `reset_index=true` или передайте `--reset-index`, иначе будет оцениваться старая коллекция Qdrant.

Параметры для перебора: `top_k`, `chunk_size`, `chunk_overlap`, `embedding_model`, `embedding_dim`, `generator_model`, `max_new_tokens`, `temperature`, `torch_dtype`, `qdrant_collection`.

Результаты сохраняются в трех форматах:

- `eval/results/ragas_YYYYMMDD_HHMMSS.json` — полный payload с samples, results и summary;
- `eval/results/ragas_YYYYMMDD_HHMMSS.csv` — табличный вывод RAGAS;
- `eval/results/ragas_YYYYMMDD_HHMMSS.md` — короткий markdown-отчет.

Метрики RAGAS являются LLM-based, поэтому при полном запуске будут загружены evaluator LLM и embeddings. По умолчанию используются те же Qwen и Octen-Embedding-0.6B через LangChain-обертки RAGAS.

## Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `RAG_API_BIND` | `0.0.0.0` | адрес публикации API в Docker |
| `RAG_API_PORT` | `8000` | внешний порт API |
| `QDRANT_HOST` | `localhost` | Qdrant для локального Python-запуска |
| `QDRANT_PORT` | `6333` | HTTP-порт Qdrant внутри приложения |
| `QDRANT_HTTP_BIND` | `127.0.0.1` | адрес публикации HTTP-порта Qdrant |
| `QDRANT_GRPC_BIND` | `127.0.0.1` | адрес публикации gRPC-порта Qdrant |
| `QDRANT_COLLECTION` | `documents` | коллекция Qdrant |
| `EMBEDDING_MODEL` | `Octen/Octen-Embedding-0.6B` | модель эмбеддингов |
| `EMBEDDING_DIM` | `1024` | размерность векторов |
| `GENERATOR_MODEL` | `Qwen/Qwen3.5-4B` | модель генерации |
| `SYSTEM_PROMPT` | пусто | системный промпт; если пусто, используется промпт по умолчанию |
| `TORCH_DTYPE` | `float16` | `auto`, `float16`, `bfloat16`, `float32` |
| `MAX_NEW_TOKENS` | `384` | максимум новых токенов |
| `TEMPERATURE` | `0.2` | температура генерации |
| `TOP_K` | `5` | сколько чанков извлекать |
| `CONTEXT_WINDOW_TURNS` | `3` | сколько последних пар вопрос/ответ хранить для уточняющих запросов |
| `CHUNK_SIZE` | `512` | размер чанка |
| `CHUNK_OVERLAP` | `64` | перекрытие чанков |
| `RAGAS_CONFIG_PATH` | `eval/ragas_config.json` | конфигурация RAGAS |
| `RAGAS_TESTSET_PATH` | `eval/testset.jsonl` | тестовый набор RAGAS |
| `RAGAS_OUTPUT_DIR` | `eval/results` | папка отчетов RAGAS |
| `RAGAS_TIMEOUT` | `900` | таймаут одной RAGAS-задачи в секундах |
| `RAGAS_MAX_WORKERS` | `1` | параллелизм RAGAS; для локального vLLM лучше начинать с 1 |
| `RAGAS_LLM_MODEL` | `Qwen/Qwen3.5-4B` | judge LLM, который обслуживает vLLM |
| `RAGAS_LLM_BASE_URL` | `http://localhost:8001/v1` | OpenAI-compatible endpoint для локального запуска Python |
| `RAGAS_LLM_API_KEY` | `local-vllm-key` | API key для vLLM |
| `RAGAS_LLM_MAX_TOKENS` | `512` | максимальный размер ответа judge LLM для RAGAS |
| `RAGAS_LLM_TIMEOUT` | `900` | HTTP timeout запросов к vLLM judge LLM в секундах |
| `RAGAS_LLM_WAIT_TIMEOUT` | `600` | сколько секунд ждать готовности `/v1/models` при старте оценки |
| `RAGAS_LLM_WAIT_INTERVAL` | `5` | интервал между проверками готовности vLLM |
| `VLLM_API_PORT` | `8001` | внешний порт OpenAI-compatible сервера vLLM |
| `VLLM_MAX_MODEL_LEN` | `4096` | максимальная длина контекста vLLM; влияет на размер KV-cache |
| `VLLM_MAX_NUM_SEQS` | `1` | максимум одновременных sequences для vLLM judge |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `4096` | верхняя граница batch tokens для vLLM |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.55` | доля GPU-памяти, которую vLLM может использовать под веса и KV-cache |

## Данные для RAG

- `./data` монтируется в контейнер как `/app/data`.
- API разрешает индексировать только пути внутри `data`, чтобы случайно не читать служебные файлы контейнера.
- модели HuggingFace кешируются в Docker volume `hf_cache`;
- данные Qdrant хранятся в Docker volume `qdrant_storage`.

## Проверка

```bash
.\.venv\Scripts\python.exe -m pytest -q
docker compose config
```
