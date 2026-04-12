import asyncio
import logging
import os
import json
import random
import itertools
from urllib.parse import urlparse, urlunparse
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from config import BASE_URL, HEADLESS_MODE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательный класс: пул платных HTTPS-прокси с ротацией и fallback
# ---------------------------------------------------------------------------
class ProxyPool:
    """
    Управляет списком платных HTTPS-прокси и выдаёт их по кругу.

    Формат каждого прокси-словаря (совместим с Playwright):
        {
            "server":   "https://host:port",   # обязательно
            "username": "login",               # если требует аутентификации
            "password": "secret",              # если требует аутентификации
        }
    """

    def __init__(self, proxies: list[dict]):
        if not proxies:
            raise ValueError("ProxyPool: список прокси не может быть пустым")
        # Проверяем, что каждый прокси содержит обязательное поле server
        for p in proxies:
            if "server" not in p:
                raise ValueError(f"ProxyPool: прокси без поля 'server': {p}")
            if not p["server"].startswith(("http://", "https://")):
                raise ValueError(
                    f"ProxyPool: поле 'server' должно начинаться с http:// или https://: {p['server']}"
                )
        self._pool = list(proxies)
        self._cycle = itertools.cycle(self._pool)
        self._failed: set[str] = set()
        logger.info("ProxyPool инициализирован: %d прокси", len(self._pool))

    def next(self) -> dict | None:
        """Возвращает следующий рабочий прокси (пропускает помеченные как сбойные)."""
        for _ in range(len(self._pool)):
            candidate = next(self._cycle)
            if candidate["server"] not in self._failed:
                logger.debug("ProxyPool: выбран прокси %s", candidate["server"])
                return candidate
        logger.error("ProxyPool: все прокси помечены как сбойные!")
        return None

    def mark_failed(self, proxy: dict) -> None:
        """Помечает прокси как нерабочий (исключает из ротации)."""
        server = proxy.get("server", "")
        self._failed.add(server)
        logger.warning("ProxyPool: прокси %s помечен как сбойный (%d/%d рабочих)",
                        server, len(self._pool) - len(self._failed), len(self._pool))

    def reset_failed(self) -> None:
        """Снимает пометку «сбойный» со всех прокси (например, при повторном запуске)."""
        self._failed.clear()
        logger.info("ProxyPool: список сбойных прокси сброшен")

    @classmethod
    def from_config(cls) -> "ProxyPool | None":
        """
        Создаёт пул из конфига.  В config.py должна быть переменная PROXY_LIST —
        список словарей.  Если переменная отсутствует или пуста — возвращает None
        (прокси не используется).

        Пример config.py:
            PROXY_LIST = [
                {
                    "server":   "https://zproxy.lum-superproxy.io:22225",
                    "username": "brd-customer-XXXXXX-zone-residential",
                    "password": "hunter2",
                },
                {
                    "server":   "https://pr.oxylabs.io:7777",
                    "username": "customer-myname",
                    "password": "secret",
                },
            ]
        """
        try:
            import config as _cfg
            proxy_list = getattr(_cfg, "PROXY_LIST", None)
            if not proxy_list:
                return None
            return cls(proxy_list)
        except Exception as e:
            logger.warning("ProxyPool.from_config: не удалось прочитать PROXY_LIST: %s", e)
            return None

# Пул реалистичных User-Agent (Chrome разных версий, Windows + Mac)
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]

# Пул типичных разрешений экрана
_VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1440, "height": 900},
    {"width": 1536, "height": 864},
    {"width": 1366, "height": 768},
]

# Stealth instance с настройками под русскую локаль
_STEALTH = Stealth(
    navigator_languages_override=("ru-RU", "ru", "en-US", "en"),
)


class RostenderSession:
    def __init__(self, auth_url, headless=True, proxy=None, screenshot_dir=None,
                 proxy_pool=None):
        self.auth_url = auth_url
        self.headless = headless
        # proxy_pool имеет приоритет над одиночным proxy
        self._proxy_pool = proxy_pool
        self.proxy = proxy_pool.next() if proxy_pool else proxy   # dict {"server","username","password"} или None
        self.screenshot_dir = screenshot_dir  # папка для скриншотов; None — скриншоты отключены
        self._stealth_cm = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    @property
    def context(self):
        return self._context

    async def _rotate_proxy(self) -> bool:
        """
        Помечает текущий прокси как сбойный и берёт следующий из пула.
        Возвращает True, если удалось переключиться; False — если пул исчерпан.
        Автоматически пересоздаёт контекст браузера с новым прокси.
        """
        if not self._proxy_pool:
            logger.warning("_rotate_proxy: пул прокси не задан, ротация невозможна")
            return False

        if self.proxy:
            self._proxy_pool.mark_failed(self.proxy)

        new_proxy = self._proxy_pool.next()
        if not new_proxy:
            logger.error("_rotate_proxy: рабочих прокси не осталось")
            return False

        logger.info("_rotate_proxy: переключение на прокси %s", new_proxy["server"])
        self.proxy = new_proxy

        # Пересоздаём контекст браузера с новым прокси
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
        _ua = random.choice(_USER_AGENTS)
        _viewport = random.choice(_VIEWPORTS)
        context_options = dict(
            user_agent=_ua,
            viewport=_viewport,
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            proxy=new_proxy,
            extra_http_headers={
                "Accept":                  "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language":         "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding":         "gzip, deflate, br",
                "Connection":              "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest":          "document",
                "Sec-Fetch-Mode":          "navigate",
                "Sec-Fetch-Site":          "none",
                "Sec-Fetch-User":          "?1",
            },
        )
        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()
        logger.info("_rotate_proxy: контекст пересоздан с прокси %s", new_proxy["server"])
        return True

    async def _screenshot(self, page, name: str) -> None:
        """Сохраняет скриншот страницы, если задана папка screenshot_dir."""
        if not self.screenshot_dir:
            return
        os.makedirs(self.screenshot_dir, exist_ok=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.screenshot_dir, f"{ts}_{name}.png")
        try:
            await page.screenshot(path=path, full_page=True)
            logger.info("Скриншот сохранён: %s", path)
        except Exception as e:
            logger.warning("Не удалось сохранить скриншот '%s': %s", name, e)

    async def _human_delay(self, min_ms: int = 300, max_ms: int = 1200) -> None:
        """Случайная пауза, имитирующая время реакции человека."""
        await self._page.wait_for_timeout(random.randint(min_ms, max_ms))

    async def _human_scroll(self, page) -> None:
        """Плавный скролл вниз по странице с нерегулярными паузами."""
        total_height = await page.evaluate("document.body.scrollHeight")
        current = 0
        while current < total_height:
            step = random.randint(250, 600)
            current = min(current + step, total_height)
            await page.evaluate(f"window.scrollTo({{top: {current}, behavior: 'smooth'}})")
            await page.wait_for_timeout(random.randint(150, 500))
        # Иногда чуть прокручиваем обратно — живой человек так и делает
        if random.random() < 0.4:
            back = random.randint(100, 350)
            await page.evaluate(f"window.scrollBy({{top: -{back}, behavior: 'smooth'}})")
            await page.wait_for_timeout(random.randint(200, 500))

    async def _human_click(self, page, locator) -> None:
        """Перемещает мышь к элементу с небольшим случайным оффсетом перед кликом."""
        box = await locator.bounding_box()
        if box:
            x = box["x"] + box["width"]  * random.uniform(0.3, 0.7)
            y = box["y"] + box["height"] * random.uniform(0.3, 0.7)
            await page.mouse.move(x, y, steps=random.randint(10, 25))
            await page.wait_for_timeout(random.randint(80, 250))
        await locator.click()

    async def _human_type(self, page, text: str) -> None:
        """
        Печатает текст с человеческими задержками между символами.
        С небольшой вероятностью делает опечатку и тут же исправляет её.
        """
        for char in text:
            if random.random() < 0.04:          # ~4% шанс опечатки
                typo = random.choice("qwertyuiopasdfghjklzxcvbnm")
                await page.keyboard.type(typo, delay=random.randint(60, 130))
                await page.wait_for_timeout(random.randint(120, 350))
                await page.keyboard.press("Backspace")
                await page.wait_for_timeout(random.randint(80, 200))
            await page.keyboard.type(char, delay=random.randint(60, 130))

    async def start(self):
        # use_async() возвращает AsyncContextManager, а не корутину —
        # входим в него вручную через __aenter__ / __aexit__.
        self._stealth_cm = _STEALTH.use_async(async_playwright())
        self._playwright = await self._stealth_cm.__aenter__()
        logger.info("Запуск браузера (headless=%s)", self.headless)

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-infobars",
            "--window-size=1920,1080",
            "--start-maximized",
            "--disable-extensions",
            "--disable-setuid-sandbox",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
            "--disable-accelerated-2d-canvas",
            "--disable-gpu",
            ],
        )

        # Случайный UA и viewport — каждый запуск выглядит как новый пользователь
        _ua       = random.choice(_USER_AGENTS)
        _viewport = random.choice(_VIEWPORTS)
        logger.info("UA: %s | viewport: %sx%s", _ua, _viewport["width"], _viewport["height"])

        context_options = dict(
            user_agent=_ua,
            viewport=_viewport,
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            extra_http_headers={
                "Accept":                  "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language":         "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding":         "gzip, deflate, br",
                "Connection":              "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest":          "document",
                "Sec-Fetch-Mode":          "navigate",
                "Sec-Fetch-Site":          "none",
                "Sec-Fetch-User":          "?1",
            },
        )
        if self.proxy:
            context_options["proxy"] = self.proxy
            logger.info("Используется прокси: %s", self.proxy.get("server"))

        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()

        # Каждый запуск — свежая сессия, сразу авторизуемся
        if self.headless:
            await self._login_with_credentials()
        else:
            await self._login_with_auth_url()

    async def _login_with_auth_url(self):
        """headless=False: открываем magic-ссылку и ждём авторизации (упрощённо, но надёжно)."""
        logger.info("Переход по ссылке авторизации: %s", self.auth_url)
        await self._page.goto(self.auth_url, wait_until="domcontentloaded")
        try:
            # Ждём элемент авторизованного пользователя (как в исходном коде)
            await self._page.wait_for_selector(".header-login__name", timeout=15000)
            logger.info("Авторизация подтверждена")
            await self._screenshot(self._page, "login_success")
        except Exception as e:
            await self._screenshot(self._page, "login_error")
            logger.error("Не удалось подтвердить авторизацию по ссылке: %s", e)
            await self.close()
            raise

    async def _login_with_credentials(self):
        """
        headless=True: заполняем форму логина с имитацией поведения человека —
        заход через главную, случайные задержки, движение мыши, опечатки.
        """
        from config import LOGIN, LOG_PASSWORD

        # --- Заход через главную, а не сразу на /login ---
        logger.info("Заход через главную страницу перед авторизацией")
        await self._page.goto("https://rostender.info/", wait_until="networkidle", timeout=60000)
        await self._human_delay(1000, 2500)
        await self._human_scroll(self._page)
        await self._human_delay(500, 1500)

        login_page_url = "https://rostender.info/login"
        logger.info("Переход на страницу входа: %s", login_page_url)
        await self._page.goto(login_page_url, wait_until="networkidle", timeout=60000)
        await self._human_delay(600, 1400)

        pwd_input = self._page.locator('input[type="password"]')
        await pwd_input.wait_for(state="visible", timeout=10000)
        await self._screenshot(self._page, "login_form")

        # --- Заполняем логин ---
        login_field = self._page.locator('input[type="text"], input[type="email"]').first
        await self._human_click(self._page, login_field)
        await self._human_delay(200, 500)
        await self._human_type(self._page, LOGIN)
        await self._page.keyboard.press("Tab")
        await self._human_delay(300, 700)

        # --- Заполняем пароль ---
        await self._human_click(self._page, pwd_input)
        await self._human_delay(200, 500)
        await self._human_type(self._page, LOG_PASSWORD)
        await self._page.keyboard.press("Tab")
        await self._human_delay(300, 800)

        # --- Галочка согласия ---
        checkbox = self._page.locator('input[type="checkbox"]').first
        if await checkbox.count() > 0 and not await checkbox.is_checked():
            await self._human_click(self._page, checkbox)
            await self._human_delay(200, 500)

        await self._human_delay(400, 900)

        # --- Клик по кнопке "Войти" ---
        # Ищем по тексту через get_by_role — надёжнее чем form button,
        # который может резолвиться в скрытые кнопки из шапки сайта
        btn_locator = self._page.get_by_role("button", name="Войти")
        await btn_locator.wait_for(state="visible", timeout=10000)

        async with self._page.expect_navigation(wait_until="networkidle", timeout=60000):
            await self._human_click(self._page, btn_locator)

        await self._page.wait_for_load_state("networkidle", timeout=15000)
        current_url = self._page.url

        if "/login" in current_url:
            await self._screenshot(self._page, "login_error")
            logger.error("Авторизация не удалась, URL: %s", current_url)
            raise RuntimeError(
                f"Авторизация не удалась — остались на /login. URL: {current_url}"
            )

        # Vue SPA после редиректа может не подхватить сессию без полной перезагрузки —
        # делаем reload и ждём появления элемента авторизованного пользователя
        logger.info("Перезагрузка страницы для инициализации Vue-сессии...")
        await self._page.reload(wait_until="domcontentloaded", timeout=30000)
        try:
            await self._page.wait_for_selector(".header-login__name", timeout=15000)
            logger.info("Авторизация подтверждена: элемент пользователя найден")
        except Exception:
            await self._screenshot(self._page, "login_not_confirmed")
            raise RuntimeError(
                "Редирект прошёл, но элемент авторизации (.header-login__name) не появился. "
                "Проверьте скриншот login_not_confirmed."
            )

        logger.info("Авторизация успешна, текущий URL: %s", self._page.url)
        await self._screenshot(self._page, "login_success")

    async def close(self):
        try:
            if self._browser:
                await self._browser.close()
        finally:
            self._browser = None
            self._context = None
            self._page = None
            if self._stealth_cm is not None:
                await self._stealth_cm.__aexit__(None, None, None)
            self._stealth_cm = None
            self._playwright = None

    async def get_tender_info(self, tender_url, download_folder):
        """
        Собирает информацию о тендере и сохраняет её в data.json.
        Для headless=False используется упрощённая логика: без скроллов, задержек,
        ожидания Vue-элементов; загрузка страницы до DOMContentLoaded.
        """
        if not self._context:
            raise RuntimeError("Сессия не инициализирована. Используйте 'async with RostenderSession(...)'.")

        page = self._page

        # --- FIX 1: убираем ?h=... токен из URL ---
        parsed = urlparse(tender_url)
        clean_url = urlunparse(parsed._replace(query=""))
        if clean_url != tender_url:
            logger.info("URL тендера очищен от параметров: %s → %s", tender_url, clean_url)

        tender_page_to_close = None
        try:
            if self.headless:
                # Диагностика: логируем куки контекста перед открытием тендера
                cookies = await self._context.cookies(["https://rostender.info"])
                auth_cookie = next((c for c in cookies if c["name"] == "is_autorized"), None)
                sessid_cookie = next((c for c in cookies if c["name"] == "PHPSESSID"), None)
                logger.info(
                    "Куки перед открытием тендера: is_autorized=%s, PHPSESSID=%s",
                    auth_cookie["value"][:8] + "..." if auth_cookie else "ОТСУТСТВУЕТ",
                    sessid_cookie["value"][:8] + "..." if sessid_cookie else "ОТСУТСТВУЕТ",
                )
                if not auth_cookie or not sessid_cookie:
                    logger.error("Критические куки авторизации отсутствуют — сессия не установлена!")

                # Клиентская навигация для сохранения состояния SPA
                await page.evaluate(f"window.location.href = '{clean_url}'")
                await page.wait_for_load_state("domcontentloaded", timeout=30000)

                # Ждём появления элемента авторизации
                try:
                    await page.wait_for_selector(".header-login__name", timeout=10000)
                    logger.info("Авторизация подтверждена после клиентской навигации")
                except Exception:
                    logger.warning(
                        "Шапка авторизации не появилась после клиентской навигации — "
                        "пробуем полный reload страницы тендера"
                    )
                    await page.reload(wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_timeout(2000)

                # Проверяем авторизацию прямо на странице тендера
                is_auth_on_page = await page.locator(".header-login__name").count() > 0
                logger.info("Авторизация на странице тендера: %s", "да" if is_auth_on_page else "НЕТ")

                await self._human_scroll(page)
                await self._screenshot(page, "tender_page")

            else:
                # --- headless=False: упрощённый переход, но с проверкой сессии ---
                logger.info("Переход на страницу тендера (упрощённый): %s", clean_url)

                # Убеждаемся, что сессия активна (кука is_autorized)
                cookies = await self._context.cookies(["https://rostender.info"])
                auth_cookie = next((c for c in cookies if c["name"] == "is_autorized"), None)
                if not auth_cookie:
                    logger.warning("Кука is_autorized отсутствует, пробуем переавторизоваться")
                    await self._login_with_auth_url()

                # Используем новую страницу, чтобы не мешать странице авторизации
                page = await self._context.new_page()
                tender_page_to_close = page  # запоминаем, чтобы закрыть после обработки
                try:
                    await page.goto(clean_url, wait_until="domcontentloaded", timeout=30000)
                except Exception as e:
                    if "ERR_ABORTED" in str(e):
                        logger.warning("Навигация прервана (ERR_ABORTED), пробуем клиентскую навигацию")
                        await page.evaluate(f"window.location.href = '{clean_url}'")
                        await page.wait_for_load_state("domcontentloaded", timeout=30000)
                    else:
                        raise

                # Короткая пауза для стабильности
                await page.wait_for_timeout(500)

            # --- Извлечение информации о тендере (общая часть) ---
            tender_info = await page.evaluate('''
                () => {
                    const getText = (selector) => {
                        const el = document.querySelector(selector);
                        return el ? el.innerText.trim() : '';
                    };
                    const getFieldByLabel = (labelText) => {
                        const labels = Array.from(document.querySelectorAll('span.tender-body__label'));
                        const label = labels.find(l => l.innerText.trim() === labelText);
                        if (label && label.nextElementSibling) {
                            return label.nextElementSibling.innerText.trim();
                        }
                        return '';
                    };
                    const title = getText('h1.tender-header__h4');
                    const numberBlock = getText('.tender-info-header-number');
                    const numberMatch = numberBlock.match(/\\d+/);
                    const number = numberMatch ? numberMatch[0] : '';
                    const statusEl = document.querySelector('.accent');
                    const status = statusEl ? statusEl.innerText.trim() : 'Неизвестен';
                    const pubBlock = getText('.tender-info-header-start_date');
                    const pubMatch = pubBlock.match(/\\d{2}\\.\\d{2}\\.\\d{2,4}/);
                    const publishDate = pubMatch ? pubMatch[0] : '';
                    const endDate = getFieldByLabel('Окончание (МСК)');
                    const customer = getFieldByLabel('Заказчик');
                    const startPrice = getFieldByLabel('Начальная цена');
                    const description = title;
                    let delivery_place = '';
                    const placeBlock = document.querySelector('.tender-body__block.n1');
                    if (placeBlock) {
                        const fieldSpan = placeBlock.querySelector('.tender-body__field');
                        if (fieldSpan) {
                            delivery_place = fieldSpan.innerText.trim();
                        }
                    }
                    if (!delivery_place) {
                        delivery_place = getFieldByLabel('Место поставки');
                    }

                    // --- Парсинг таблицы позиций ---
                    const positions = [];
                    const positionsContainer = document.querySelector('#positions');
                    if (positionsContainer) {
                        const rows = positionsContainer.querySelectorAll('.table-positions-raw');
                        for (const row of rows) {
                            if (row.classList.contains('table-positions-header') || row.classList.contains('itog-raw')) continue;

                            const numberCell = row.querySelector('.cell-number');
                            if (!numberCell) continue;
                            const number = numberCell.innerText.trim();

                            const nameCell = row.querySelector('.cell-name');
                            const name = nameCell ? nameCell.innerText.trim() : '';

                            let okpd2_code = '', ktru_code = '';
                            const classifierCell = row.querySelector('.cell-classifier');
                            if (classifierCell) {
                                const classifierDivs = classifierCell.querySelectorAll('.classifier');
                                classifierDivs.forEach(div => {
                                    const text = div.innerText.trim();
                                    if (text.includes('ОКПД2:')) {
                                        okpd2_code = text.replace('ОКПД2:', '').trim();
                                    } else if (text.includes('КТРУ:')) {
                                        ktru_code = text.replace('КТРУ:', '').trim();
                                    }
                                });
                            }

                            let quantity = '', unit = '';
                            const quantityCell = row.querySelector('.cell-quantity');
                            if (quantityCell) {
                                const quantityText = quantityCell.innerText.trim();
                                const match = quantityText.match(/([\\d\\s]+?)\\s*\\(([^)]+)\\)/);
                                if (match) {
                                    quantity = match[1].trim().replace(/\\s/g, '');
                                    unit = match[2].trim();
                                } else {
                                    quantity = quantityText;
                                }
                            }

                            const priceCell = row.querySelector('.cell-price');
                            const price = priceCell ? priceCell.innerText.trim() : '';
                            const sumCell = row.querySelector('.cell-sum');
                            const sum = sumCell ? sumCell.innerText.trim() : '';

                            positions.push({
                                number,
                                name,
                                okpd2_code,
                                ktru_code,
                                quantity,
                                unit,
                                price,
                                sum
                            });
                        }
                    }

                    return {
                        title,
                        number,
                        status,
                        publish_date: publishDate,
                        end_date: endDate,
                        customer,
                        start_price: startPrice,
                        description,
                        delivery_place,
                        positions,
                        url: window.location.href
                    };
                }
            ''')

            # --- Проверка статуса тендера (не завершён) ---
            if tender_info['status'] in ("Завершён", "Отменён"):
                logger.info("Тендер завершён/отменен, файлы не собираются")
                return tender_info

            # --- Сбор ссылок на файлы документации (общая часть) ---
            try:
                await page.wait_for_selector(".tender-files", timeout=15000)
            except Exception:
                logger.warning("Блок документации не найден")
                tender_info["files"] = []
                os.makedirs(download_folder, exist_ok=True)
                data_path = os.path.join(download_folder, "data.json")
                with open(data_path, "w", encoding='utf-8') as f:
                    json.dump(tender_info, f, ensure_ascii=False, indent=4)
                return tender_info

            file_links = await page.locator(".tender-files a.icon-download-gray").all()
            files_info = []
            for link in file_links:
                href = await link.get_attribute("href")
                title = await link.evaluate("""
                    (el) => {
                        const li = el.closest('li');
                        if (!li) return 'Без названия';
                        const titleEl = li.querySelector('.title');
                        return titleEl ? titleEl.innerText.trim() : 'Без названия';
                    }
                """)
                if not href:
                    continue
                if href.startswith(("http://", "https://")):
                    file_url = href
                elif href.startswith("/"):
                    file_url = BASE_URL.rstrip("/") + href
                else:
                    continue
                files_info.append({"name": title, "url": file_url})

            logger.info("Найдено файлов на странице: %d", len(files_info))
            tender_info["files"] = files_info

            # --- Сохранение итоговой информации ---
            os.makedirs(download_folder, exist_ok=True)
            data_path = os.path.join(download_folder, "data.json")
            with open(data_path, "w", encoding='utf-8') as f:
                json.dump(tender_info, f, ensure_ascii=False, indent=4)

            return tender_info

        except Exception:
            await self._screenshot(page, "tender_error")
            raise
        finally:
            # Закрываем вкладку тендера (только при headless=False, когда создавали новую страницу)
            if tender_page_to_close is not None:
                try:
                    await tender_page_to_close.close()
                    logger.debug("Вкладка тендера закрыта")
                except Exception as e:
                    logger.warning("Не удалось закрыть вкладку тендера: %s", e)


async def get_tender_data(auth_url, tender_url, download_folder, headless=True, proxy=None,
                          screenshot_dir=None, proxy_pool=None):
    """
    Упрощённая функция для получения данных тендера без скачивания файлов.

    proxy_pool (ProxyPool | None): пул платных прокси с автоматической ротацией.
        Если передан — имеет приоритет над одиночным proxy.
    proxy (dict | None): одиночный прокси {"server": "...", "username": "...", "password": "..."}.
        Используется, только если proxy_pool не передан.
    """
    async with RostenderSession(
        auth_url, headless=headless, proxy=proxy,
        screenshot_dir=screenshot_dir, proxy_pool=proxy_pool,
    ) as session:
        return await session.get_tender_info(tender_url, download_folder)


async def main():
    from config import AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER
    import config as _cfg

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    screenshot_dir = getattr(_cfg, "SCREENSHOT_DIR", None)

    # --- Инициализация прокси ---
    # Приоритет: PROXY_LIST (пул с ротацией) > PROXY (одиночный) > None (без прокси)
    proxy_pool = ProxyPool.from_config()
    if proxy_pool:
        proxy = None
        logger.info("Используется пул прокси (%d штук)", len(proxy_pool._pool))
    else:
        proxy = getattr(_cfg, "PROXY", None)
        if proxy:
            logger.info("Используется одиночный прокси: %s", proxy.get("server"))
        else:
            logger.info("Прокси не настроен — прямое соединение")

    tender_data = await get_tender_data(
        AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER,
        headless=HEADLESS_MODE, proxy=proxy,
        proxy_pool=proxy_pool, screenshot_dir=screenshot_dir,
    )
    logger.info("Сбор данных завершён. Информация сохранена в %s", DOWNLOAD_FOLDER)
    if "files" in tender_data:
        for f in tender_data["files"]:
            logger.info("Файл: %s -> %s", f["name"], f["url"])


if __name__ == "__main__":
    asyncio.run(main())