import imaplib
import email
import re
import logging
import email.utils
import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from config import (
    IMAP_SERVER,
    EMAIL_ACCOUNT,
    PASSWORD,
    ALLOWED_SENDERS,
    SMTP_SERVER,
    SMTP_ACCOUNT,
    SMTP_PORT,
    SMTP_PASSWORD,
    MAIL_FROM,
    MAIL_TO

)

def extract_links_from_html(html):
    """Возвращает список URL из HTML для ссылок rostender.info/tender/."""
    soup = BeautifulSoup(html, 'lxml')
    urls = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('https://rostender.info/tender/'):
            urls.append(href)
    return urls

def extract_links_from_text(text):
    """Возвращает список URL из plain‑text для ссылок rostender.info/tender/."""
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
    all_urls = url_pattern.findall(text)
    filtered = []
    for url in all_urls:
        if url.startswith('https://rostender.info/tender/'):
            filtered.append(url)
    return filtered

def process_email_body(msg):
    """Извлекает все ссылки из письма, возвращает список URL."""
    urls = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get("Content-Disposition") == "attachment":
                continue
            try:
                body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                continue
            content_type = part.get_content_type()
            if content_type == "text/plain":
                urls.extend(extract_links_from_text(body))
            elif content_type == "text/html":
                urls.extend(extract_links_from_html(body))
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
        if msg.get_content_type() == "text/plain":
            urls.extend(extract_links_from_text(body))
        elif msg.get_content_type() == "text/html":
            urls.extend(extract_links_from_html(body))
    return urls

def fetch_links_from_emails(save_to_file=None):
    """
    Подключается к почте, ищет непрочитанные письма от разрешённых отправителей,
    извлекает ссылки на тендеры rostender.info/tender/.
    
    Параметры:
        save_to_file: путь к файлу для сохранения ссылок (например "rostender_links_unseen.txt").
                      Если None — не сохранять в файл.
    
    Возвращает:
        list[str]: список уникальных URL тендеров.
    """
    logger.info("Подключение к IMAP-серверу %s", IMAP_SERVER)
    imap = imaplib.IMAP4_SSL(IMAP_SERVER)
    imap.login(EMAIL_ACCOUNT, PASSWORD)
    imap.select("inbox")

    search_criteria = 'UNSEEN'
    result, data = imap.uid('search', None, search_criteria)
    if result != 'OK':
        logger.error("Ошибка поиска писем по критерию UNSEEN")
        imap.logout()
        return []

    uids = data[0].split()
    logger.info("Найдено непрочитанных писем: %s", len(uids))

    all_urls = []

    for uid in uids:
        result, msg_data = imap.uid('fetch', uid, '(RFC822)')
        if result != 'OK':
            logger.warning("Не удалось получить письмо uid=%s", uid.decode() if isinstance(uid, bytes) else uid)
            continue
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        from_header = msg.get('From', '')
        real_name, real_email = email.utils.parseaddr(from_header)
        if real_email not in ALLOWED_SENDERS:
            logger.debug("Пропущено письмо от %s (не в списке разрешённых)", real_email)
            continue

        urls = process_email_body(msg)
        if urls:
            all_urls.extend(urls)
            logger.debug("Из письма от %s извлечено ссылок: %s", real_email, len(urls))

        imap.uid('STORE', uid, '+FLAGS', '\\Seen')

    unique_urls = list(set(all_urls))
    if save_to_file:
        with open(save_to_file, "w", encoding='utf-8') as f:
            for url in unique_urls:
                f.write(url + "\n")
        logger.info("Ссылки сохранены в файл: %s", save_to_file)

    logger.info("Найдено уникальных ссылок на тендеры: %s", len(unique_urls))
    imap.close()
    imap.logout()
    return unique_urls

def send_files_via_email(file_paths, subject="Отчёт по тендерам"):
    """
    Отправляет одно письмо с несколькими вложениями.
    file_paths: список путей к файлам для прикрепления.
    """
    # Создаём контейнер письма
    msg = MIMEMultipart()
    msg['From'] = MAIL_FROM
    msg['To'] = MAIL_TO
    msg['Subject'] = subject

    # Текст письма (можно добавить описание)
    body = "Сформированы отчёты по тендерам во вложениях."
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # Прикрепляем каждый файл
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден, пропускаем.")
            continue
        with open(file_path, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            filename = os.path.basename(file_path)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename="{filename}"'
            )
            msg.attach(part)

    # Отправка
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_ACCOUNT, SMTP_PASSWORD)
            server.send_message(msg)
        print(f"Письмо с вложениями успешно отправлено на {MAIL_TO}")
    except Exception as e:
        print(f"Ошибка при отправке письма: {e}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Получаем ссылки из непрочитанных писем
    links = fetch_links_from_emails(save_to_file="rostender_links_unseen.txt")

    # Здесь, после получения ссылок, обычно происходит парсинг тендеров
    # и генерация HTML-таблицы (например, в файл "tenders_summary.html").
    # Предположим, что такой файл уже создан другим процессом.
    # Попробуем отправить его, если он существует.
    html_file = "tenders_summary.html"
    if os.path.exists(html_file):
        send_files_via_email([html_file])
    else:
        logger.info("Файл %s не найден, отправка пропущена.", html_file)

if __name__ == "__main__":
    main()