import asyncio
import logging
import os
import json
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from config import BASE_URL

logger = logging.getLogger(__name__)


class RostenderSession:
    def __init__(self, auth_url, headless=False):
        self.auth_url = auth_url
        self.headless = headless
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
        self._playwright = await async_playwright().start()
        logger.info("Запуск браузера (headless=%s)", self.headless)
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()

        logger.info("Переход по ссылке авторизации: %s", self.auth_url)
        await self._page.goto(self.auth_url)

        try:
            await self._page.wait_for_selector(".header-login__name", timeout=10000)
            logger.info("Авторизация успешна")
        except Exception as e:
            logger.error("Не удалось подтвердить авторизацию: %s", e)
            await self.close()
            raise

    async def close(self):
        try:
            if self._browser:
                await self._browser.close()
        finally:
            self._browser = None
            self._context = None
            self._page = None
            if self._playwright:
                await self._playwright.stop()
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
            await page.goto(tender_url)

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
                # Всё равно сохраняем информацию
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


async def get_tender_data(auth_url, tender_url, download_folder, headless=False):
    """
    Упрощённая функция для получения данных тендера без скачивания файлов.
    """
    async with RostenderSession(auth_url, headless=headless) as session:
        return await session.get_tender_info(tender_url, download_folder)


async def main():
    from config import AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    tender_data = await get_tender_data(AUTH_URL, TENDER_URL, DOWNLOAD_FOLDER, headless=False)
    logger.info("Сбор данных завершён. Информация сохранена в %s", DOWNLOAD_FOLDER)
    # Пример вывода полученных ссылок
    if "files" in tender_data:
        for f in tender_data["files"]:
            logger.info("Файл: %s -> %s", f["name"], f["url"])


if __name__ == "__main__":
    asyncio.run(main())