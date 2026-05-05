# Деплой Tender Web App в Docker

Веб-приложение упаковано в один Docker-образ. Данные (jsonl-отчёты) монтируются как том в read-only режиме — пересобирать образ при появлении новых отчётов не нужно.

---

## TL;DR

```bash
cp .env.example .env             # подправьте TENDERS_DATA_DIR и HOST_PORT
docker compose up -d --build     # запустит на http://<host>:5000
```

---

## Что в комплекте

| Файл | Назначение |
|---|---|
| `Dockerfile` | Образ приложения (Python 3.12 slim + gunicorn) |
| `.dockerignore` | Что не тащить в образ |
| `docker-compose.yml` | Деплой одной командой (без прокси) |
| `docker-compose.nginx.yml` | Деплой с nginx как обратным прокси |
| `nginx/nginx.conf` | Конфиг nginx |
| `.env.example` | Шаблон переменных окружения |
| `requirements.txt` | Python-зависимости (Flask + gunicorn) |

---

## Требования

- **Docker Engine** ≥ 20.10
- **Docker Compose** v2 (плагин `docker compose`, не legacy `docker-compose`)
- Доступ к каталогу с jsonl-отчётами на хосте

---

## Шаг 1. Настройте окружение

Скопируйте шаблон и отредактируйте:

```bash
cp .env.example .env
```

В `.env` укажите:

- **`TENDERS_DATA_DIR`** — путь на **хосте** до каталога с файлами `all_tenders_data_*.jsonl`. На production-сервере используйте абсолютный путь (`/srv/tenders/jsonl`), при локальной разработке достаточно относительного (`../storage/tenders/jsonl`).
- **`HOST_PORT`** — на каком порту хоста будет доступно приложение (по умолчанию 5000).
- **`HTTP_PORT`** — то же, но для варианта с nginx (по умолчанию 80).
- **`TZ`** — часовой пояс контейнера (влияет только на логи).

> Каталог монтируется в **read-only** (`:ro`) — приложение не пишет в данные, только читает.

---

## Шаг 2. Запуск

### Вариант А. Без nginx (быстрый старт, локальная сеть)

```bash
docker compose up -d --build
```

Откройте `http://<адрес-хоста>:5000/`.

Проверить статус:
```bash
docker compose ps
docker compose logs -f tender-web
```

### Вариант Б. С nginx (рекомендуется для production)

```bash
docker compose -f docker-compose.nginx.yml up -d --build
```

Откройте `http://<адрес-хоста>/` (порт 80).

Что даёт nginx:
- gzip-сжатие ответов (быстрее загружается интерфейс),
- кеширование статики,
- единая точка входа для добавления TLS (см. ниже),
- порт 80 наружу, gunicorn при этом не торчит в интернет.

---

## Шаг 3. Обновление данных

**Новый jsonl-отчёт** — просто положите его в смонтированный каталог. Перезапуск **не нужен**:

```bash
cp all_tenders_data_20260507_120000.jsonl /srv/tenders/jsonl/
```

При следующем обновлении страницы (F5) отчёт появится в выпадающем списке.

**Обновление кода приложения:**

```bash
git pull
docker compose up -d --build
```

---

## Шаг 4. Управление

```bash
# Остановить
docker compose down

# Перезапустить
docker compose restart tender-web

# Посмотреть логи
docker compose logs -f tender-web

# Войти внутрь контейнера
docker compose exec tender-web /bin/bash

# Узнать состояние healthcheck
docker inspect --format='{{.State.Health.Status}}' tender-web
```

---

## Деплой с TLS (HTTPS)

Самый простой путь — поставить **Caddy** или **Traefik** перед `tender-web` (они автоматически получают сертификаты Let's Encrypt). Альтернатива: certbot + nginx.

Минимальный пример **Caddy** (создайте `Caddyfile`):

```caddy
tenders.example.com {
    reverse_proxy tender-web:5000
}
```

И добавьте сервис в compose:

```yaml
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
    depends_on:
      - tender-web
    networks:
      - tender-net

volumes:
  caddy_data:
  caddy_config:
```

DNS-запись `tenders.example.com` должна указывать на ваш сервер — Caddy сам выпустит сертификат.

---

## Безопасность

- Контейнер запускается от **непривилегированного пользователя** (`uid=1000`, не root).
- Каталог с данными смонтирован в **read-only** (`:ro`).
- Если приложение должно быть доступно только из локальной сети — ограничьте порт через bind-адрес. В `docker-compose.yml`:
  ```yaml
  ports:
    - "127.0.0.1:5000:5000"   # доступ только с самого хоста
    # или
    - "192.168.1.10:5000:5000"  # только с конкретного интерфейса
  ```
- Не храните чувствительные данные в jsonl-каталоге — он отдаётся всем, у кого есть URL.
- При выставлении наружу обязательно поставьте TLS (см. выше).

---

## Ресурсы и масштабирование

- Базовый расход: **~80 МБ RAM** на контейнер (gunicorn с 2 воркерами × 4 потоками).
- Для увеличения параллелизма правьте CMD в `Dockerfile`:
  - `--workers N` — рекомендуется `2 × CPU + 1`.
  - `--threads M` — по 2–4 на воркера.
- Для ограничения ресурсов добавьте в compose:
  ```yaml
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 512M
  ```

---

## Диагностика

| Симптом | Что проверить |
|---|---|
| `docker compose up` падает на сборке | `docker compose build --no-cache` для чистой сборки. Версия Docker ≥ 20.10. |
| Контейнер запущен, но 502 от nginx | Жив ли `tender-web`: `docker compose logs tender-web`. Healthcheck: `docker inspect tender-web`. |
| `Нет отчётов в storage/tenders/jsonl` в UI | Том смонтирован? `docker compose exec tender-web ls /data/jsonl`. Если пусто — проверьте `TENDERS_DATA_DIR` в `.env`. |
| Страница открывается, но без стилей | Проверьте, что папка `static/` попала в образ: `docker compose exec tender-web ls /app/static`. |
| Нужны новые отчёты, но они не видны | Том в read-only — добавьте файлы на хосте, не внутри контейнера. F5 в браузере. |

---

## Структура каталогов на сервере (рекомендуемая)

```
/srv/
├── tender-web/              ← клонированный репозиторий (этот web_app)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .env                 ← ваши настройки
│   └── ...
└── tenders/
    └── jsonl/               ← TENDERS_DATA_DIR=/srv/tenders/jsonl
        ├── all_tenders_data_20260410_124943.jsonl
        └── ...
```
