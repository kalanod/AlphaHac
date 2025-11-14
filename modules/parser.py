import re
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Parser:
    def _clean_page_content(self, page_title: str, page_content: str) -> str:
        
        def preprocess_content(content):
            footer_patterns = [
                r'©.*?АО «Альфа-Банк».*?браузере',
                r'АО «Альфа-Банк».*?персональных данных',
                r'\+7\s?\d{3}\s?\d{3}\s?\d{2}\s?\d{2}',
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
                r'Лучший.*?по версии.*?',
                r'Свяжитесь с нами.*?вопросы',
                r'Хотите получить больше информации\?',
                r'Генеральная лицензия.*?\.',
                r'Ул\.\s?[А-Я].*?\d+',
                r'Частые вопросы.*',
            ]
            for pattern in footer_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
            
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            content = re.sub(r'\t+', ' ', content)
            content = re.sub(r'[|]{2,}', '', content)
            content = re.sub(r'[-=]{3,}', '', content)
            content = re.sub(r'^- •\s*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'- •\s*\n\s*([А-Яа-я])', r'• \1', content)
            content = re.sub(r'•\s*\n\s*([А-Яа-я])', r'• \1', content)
            content = re.sub(r'[🅰️✅💰🏦📱💳🔒⭐🎯📊]', '', content)
            
            nav_patterns = [
                r'Главная\s*>\s*',
                r'Перейти к содержимому',
                r'Меню\s*\n',
                r'Выберите город',
                r'Список отделений',
                r'Больше.*?\n',
                r'Подробнее.*?\n'
            ]
            
            for pattern in nav_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                
            return content.strip()
        
        def smart_line_filter(content):
            lines = content.split('\n')
            filtered_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                words = line.split()
                word_count = len(words)
                
                if word_count <= 2:
                    continue
                
                if word_count <= 4:
                    has_verb = bool(re.search(r'\b(является|содержит|включает|предоставляет|обеспечивает|позволяет|составляет|достигает|превышает|равен|имеет|может|должен|будет|был|есть|было|были|быть|иметь|делать|работает|функционирует|открывает|закрывает|оплачивает|переводит|зачисляет|начисляет|списывает|предлагает|требует|гарантирует|осуществляет|выполняет|производит|создает|разрабатывает|внедряет|поддерживает|обслуживает)\b', line, re.IGNORECASE))
                    has_numbers = bool(re.search(r'\d', line))
                    has_conditions = bool(re.search(r'\b(до|от|в|на|для|с|при|через|по|под|над|между|среди|внутри|около|примерно|более|менее|свыше|рублей|процентов|годовых|месяцев|дней|лет|часов|минут|условия|тарифы|ставка|размер|сумма|лимит|комиссия|льгота|скидка|бонус|кэшбэк)\b', line, re.IGNORECASE))
                    
                    if not (has_verb or has_numbers or has_conditions):
                        continue
                
                title_patterns = [
                    r'^(Брокерские услуги|Инвестиционное консультирование|Альтернативные инвестиции|Страховые и пенсионные программы|Альфа‑Капитал|Индивидуальные решения|Депозиты и накопительные счета|Банковские карты|Драгоценные металлы и камни|Сделки через аккредитив или депозитарий|Биометрический депозитарий|Кредитование|Платёжные аксессуары|Перевозка, доставка и инкассация наличных|Своя служба инкассации)$',
                    r'^(Frank RG|Wealth Navigator|Euromoney|По версии).*$',
                    r'^[А-Я][а-я]+\s+[а-я]+$',
                    r'^[А-Я][а-я]+\s+[а-я]+\s+[а-я]+$'
                ]
                
                should_remove = any(re.match(pattern, line) for pattern in title_patterns)
                if should_remove:
                    continue
                
                if re.match(r'^[A-Z][a-zA-Z\s]*[A-Z]$', line) and word_count <= 3:
                    continue
                
                has_concrete_info = bool(re.search(r'(\d+|рублей|процент|год|месяц|день|час|условие|тариф|ставка|размер|сумма|лимит|комиссия|возможность.*?[а-я]+|для.*?[а-я]+|в.*?[а-я]+|на.*?[а-я]+|до.*?\d|от.*?\d|свыше.*?\d|более.*?\d|менее.*?\d)', line, re.IGNORECASE))
                has_action_description = bool(re.search(r'(открытие|создание|предоставление|обслуживание|управление|структурирование|инвестирование|страхование|кредитование|депонирование).*?[а-я]+', line, re.IGNORECASE))
                
                if word_count >= 5 and not (has_concrete_info or has_action_description):
                    has_basic_info = bool(re.search(r'\b(приложении|платформе|фондах|компаний|активов|продуктов|счетах|валютах|картах|слитки|сделок|хранилище|жильё|кольца|наличных|инкассации)\b', line, re.IGNORECASE))
                    if not has_basic_info:
                        continue
                
                filtered_lines.append(line)
            
            return '\n'.join(filtered_lines)
        
        preprocessed = preprocess_content(page_content)
        return smart_line_filter(preprocessed)

    def parse_train(self) -> List[Document]:
        df_websites = pd.read_csv('res/websites.csv')
        
        df_websites['text'] = df_websites.apply(
            lambda row: self._clean_page_content(row['title'], row['text']), 
            axis=1
        )
        
        loader = DataFrameLoader(df_websites, page_content_column='text')
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        
        return texts

    def parse_questions(self):
        return []
