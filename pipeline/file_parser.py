import asyncio
import logging
import os
import json
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from config import BASE_URL, HEADLESS_MODE

logger = logging.getLogger(__name__)

# Realistic user agent matching a common Chrome version on Windows
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Stealth instance с настройками под русскую локаль
_STEALTH = Stealth(
    navigator_languages_override=("ru-RU", "ru", "en-US", "en"),
)


class RostenderSession:
    def __init__(self, auth_url, headless=True, proxy=None):
        self.auth_url = auth_url
        self.headless = headless
        self.proxy = proxy  # dict вида {"server": "...", "username": "...", "password": "..."} или None
        self._stealth_cm = None   # хранит async-контекстный менеджер use_async()
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
            ],
        )

        context_options = dict(
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            extra_http_headers={
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            },
        )
        if self.proxy:
            context_options["proxy"] = self.proxy
            logger.info("Используется прокси: %s", self.proxy.get("server"))

        self._context = await self._browser.new_context(**context_options)

        # Стелс применяется автоматически ко всем страницам через _STEALTH.use_async()
        self._page = await self._context.new_page()

        if self.headless:
            await self._login_with_credentials()
        else:
            await self._login_with_auth_url()

    async def _login_with_auth_url(self):
        """headless=False: open the magic auth link directly (original behaviour)."""
        logger.info("Переход по ссылке авторизации: %s", self.auth_url)
        await self._page.goto(self.auth_url, wait_until="domcontentloaded")
        try:
            await self._page.wait_for_selector(".header-login__name", timeout=60000)
            logger.info("Авторизация по ссылке успешна")
        except Exception as e:
            logger.error("Не удалось подтвердить авторизацию по ссылке: %s", e)
            await self.close()
            raise

    async def _login_with_credentials(self):
        """
        headless=True: intercept the XHR/fetch login call to discover the real
        API endpoint & payload format, then fire it directly via requests (bypassing
        the Vue form entirely), and inject the returned session cookies into Playwright.
        """
        import aiohttp
        from config import LOGIN, LOG_PASSWORD

        login_page_url = "https://rostender.info/login"
        logger.info("Переход на страницу входа: %s", login_page_url)

        # --- Step 1: intercept the real login API call ---
        captured = {}

        async def handle_request(route, request):
            url = request.url
            # The login XHR will POST to something like /api/... or /auth/...
            if request.method == "POST" and any(
                kw in url for kw in ["/api/", "/auth/", "/login", "/user/"]
            ):
                logger.info("Перехвачен POST-запрос: %s", url)
                captured["url"] = url
                captured["headers"] = dict(request.headers)
                captured["post_data"] = request.post_data
            await route.continue_()

        await self._page.route("**/*", handle_request)
        await self._page.goto(login_page_url, wait_until="domcontentloaded")

        pwd_input = self._page.locator('input[type="password"]')
        await pwd_input.wait_for(state="visible", timeout=10000)

        # --- Step 2: fill form using focus+keyboard to maximise Vue compatibility ---
        login_field = self._page.locator('input[type="text"], input[type="email"]').first
        await login_field.focus()
        await self._page.keyboard.type(LOGIN, delay=80)
        await self._page.keyboard.press("Tab")

        await pwd_input.focus()
        await self._page.keyboard.type(LOG_PASSWORD, delay=80)
        await self._page.keyboard.press("Tab")

        # Tick consent checkbox
        checkbox = self._page.locator('input[type="checkbox"]').first
        if not await checkbox.is_checked():
            await checkbox.check()

        await self._page.wait_for_timeout(400)

        # --- Step 3: submit and capture the outgoing request ---
        async with self._page.expect_request(
            lambda r: r.method == "POST" and any(
                kw in r.url for kw in ["/api/", "/auth/", "/login", "/user/"]
            ),
            timeout=10000
        ) as req_info:
            # Click the Войти button inside the login form
            await self._page.evaluate("""
                () => {
                    const pwd = document.querySelector('input[type="password"]');
                    const form = pwd && pwd.closest('form');
                    const btn = form && form.querySelector('button[type="submit"], button');
                    if (btn) btn.click();
                }
            """)

        intercepted_req = await req_info.value
        api_url      = intercepted_req.url
        post_data    = intercepted_req.post_data
        req_headers  = intercepted_req.headers
        logger.info("Логин API URL: %s", api_url)
        logger.info("POST payload:  %s", post_data)

        # --- Step 4: wait for redirect / response ---
        try:
            await self._page.wait_for_url(
                lambda url: "/login" not in url,
                timeout=20000,
            )
        except Exception:
            pass  # might already have navigated

        await self._page.wait_for_load_state("networkidle", timeout=15000)
        current_url = self._page.url

        if "/login" in current_url:
            logger.error("Авторизация не удалась, URL: %s", current_url)
            raise RuntimeError(
                f"Авторизация не удалась — остались на /login. URL: {current_url}"
            )

        logger.info("Авторизация успешна, текущий URL: %s", current_url)

        # Remove route so it doesn't affect subsequent pages
        await self._page.unroute("**/*", handle_request)

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
        Возвращает словарь с данными тендера, включая поле 'files' со списком
        ссылок на документацию.
        """
        if not self._context:
            raise RuntimeError("Сессия не инициализирована. Используйте 'async with RostenderSession(...)'.")

        page = await self._context.new_page()
        try:
            logger.info("Переход на страницу тендера: %s", tender_url)
            await page.goto(tender_url, wait_until="domcontentloaded")

            # --- Извлечение информации о тендере ---
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

            # --- Сбор ссылок на файлы документации ---
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

        finally:
            await page.close()


async def get_tender_data(auth_url, tender_url, download_folder, headless=True, proxy=None):
    """
    Упрощённая функция для получения данных тендера без скачивания файлов.
    """
    async with RostenderSession(auth_url, headless=headless, proxy=proxy) as session:
        return await session.get_tender_info(tender_url, download_folder)


async def main():
    from config import AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER, PROXY
    import importlib, config as _cfg
    proxy = getattr(_cfg, "PROXY", None)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    tender_data = await get_tender_data(AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER, headless=HEADLESS_MODE, proxy=proxy)
    logger.info("Сбор данных завершён. Информация сохранена в %s", DOWNLOAD_FOLDER)
    if "files" in tender_data:
        for f in tender_data["files"]:
            logger.info("Файл: %s -> %s", f["name"], f["url"])


if __name__ == "__main__":
    asyncio.run(main())