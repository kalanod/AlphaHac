import re
import os
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Parser:
    def __init__(self, path):
        self.questions_path = os.path.join(path, 'questions_clean.csv')
        self.websites_path = os.path.join(path, 'websites.csv')
    
    def _clean_page_content(self, page_title: str, page_content: str) -> str:
        content = self._basic_cleanup(page_content)
        content = self._remove_footers_and_navigation(content)
        content = self._extract_useful_content(content, page_title)
        return content

    def _basic_cleanup(self, text: str) -> str:
        text = text.replace('\xa0', ' ')
        text = text.replace('\u00a0', ' ')
        text = text.replace('\u2009', ' ')
        text = text.replace('\u202f', ' ')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '-')
        text = text.replace('\u2010', '-')
        text = text.replace('\u2011', '-')
        
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"
            u"\u3030"
            "🚀📚💰🏦📱💳🔒⭐🎯📊✅🅰️"
            "]+", flags=re.UNICODE)
        
        text = emoji_pattern.sub('', text)
        text = re.sub(r'^[\s\.,;:\-\|]+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[-]{2,}', '-', text)
        text = re.sub(r'[,]{2,}', ',', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        
        return text.strip()

    def _remove_footers_and_navigation(self, text: str) -> str:
        text = re.sub(r'©.*?$', '', text, flags=re.DOTALL)
        text = self._remove_navigation_menus(text)
        
        footer_patterns = [
            r'Генеральная лицензия.*?$',
            r'\+7\s*\d{3}\s*\d{3}[-\s]*\d{2}[-\s]*\d{2}.*?$',
            r'[a-zA-Z0-9._%+-]+@alfabank\.ru.*?$',
            r'Ул\.\s*Каланчевская.*?$',
            r'(Лучший|По версии|Frank RG|Wealth Navigator|Euromoney).*?$',
            r'Хотите получить больше информации\?.*?$',
            r'Свяжитесь с нами.*?$',
            r'Позвоните нам.*?$',
            r'Частые вопросы\s*$',
            r'\d+[,\.]\d+\s*[КМГ]б\s*$',
            r'Архив\s*$',
        ]
        
        for pattern in footer_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text

    def _remove_navigation_menus(self, text: str) -> str:
        sentences = re.split(r'[.!?]\s+', text)
        cleaned_sentences = []
        
        for sentence in sentences:
            if not self._is_navigation_menu(sentence):
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences)

    def _is_navigation_menu(self, text: str) -> bool:
        words = text.split()
        
        if len(words) < 5:
            return False
        
        capitalized_count = 0
        navigation_keywords = 0
        
        nav_keywords = {
            'образование', 'магистратура', 'магистратуры', 'кампус', 'курсы', 'гранты', 
            'стипендии', 'мероприятия', 'карьера', 'партнёрства', 'поддержка', 
            'услуги', 'продукты', 'тарифы', 'условия', 'информация', 'контакты',
            'вакансии', 'амбассадоры', 'менторство', 'стажировки', 'фест',
            'экскурсии', 'митап', 'hack', 'school', 'lab', 'digital'
        }
        
        for word in words:
            if word and word[0].isupper():
                capitalized_count += 1
            
            if word.lower() in nav_keywords:
                navigation_keywords += 1
        
        capitalized_ratio = capitalized_count / len(words)
        
        if (capitalized_ratio > 0.6 and navigation_keywords >= 2) or \
           (capitalized_ratio > 0.8) or \
           (navigation_keywords >= 5):
            return True
        
        short_capitalized = sum(1 for word in words if word and word[0].isupper() and len(word) <= 8)
        if short_capitalized >= len(words) * 0.7 and len(words) > 10:
            return True
        
        return False

    def _extract_useful_content(self, text: str, title: str) -> str:
        text = self._remove_navigation_menus(text)
        
        lines = text.split('\n')
        useful_content = []
        
        if title and len(title.split()) > 3:
            clean_title = self._clean_title(title)
            if clean_title:
                useful_content.append(f"Тема: {clean_title}")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if self._is_navigation_menu(line):
                continue
            
            line = self._clean_line(line)
            if not line:
                continue
            
            if len(line.split()) < 4:
                continue
            
            if self._is_unwanted_line(line):
                continue
            
            formatted_line = self._format_line(line)
            if formatted_line:
                useful_content.append(formatted_line)
        
        result = self._post_process_content(useful_content)
        return result

    def _clean_title(self, title: str) -> str:
        title = re.sub(r'Альфа-Банк\s*-\s*', '', title)
        title = re.sub(r'–\s*Альфа-Банк.*$', '', title)
        title = re.sub(r'[🚀📚💰🏦📱💳🔒⭐🎯📊✅]', '', title)
        title = title.replace('\xa0', ' ')
        title = re.sub(r'\s+', ' ', title).strip()
        
        generic_titles = ['дневник', 'скидки', 'правила', 'главная']
        if any(generic in title.lower() for generic in generic_titles):
            return None
        
        return title if len(title) > 10 else None

    def _clean_line(self, line: str) -> str:
        line = re.sub(r'^[\s\.,;:\-\|\>\<\=\+]+', '', line)
        line = re.sub(r'[\s\.,;:\-\|\>\<\=\+]+$', '', line)
        line = re.sub(r'\s+', ' ', line)
        
        return line.strip()

    def _is_unwanted_line(self, line: str) -> bool:
        if re.search(r'^[А-Я][а-я]+\s+[А-Я][а-я]+,\s*[А-Я]', line):
            return True
        
        unwanted_patterns = [
            r'^(Главная|Меню|Больше|Подробнее|Список|Выберите|Карта сайта|Поддержка)',
            r'(USD|EUR|CNY).*\d+',
            r'\d+[,\.]\d+\s*[КМГ]б',
            r'^[\|\-\=\+\>\<\s]+$',
            r'^\+7\s*\d{3}',
            r'^[a-zA-Z0-9._%+-]+@',
            r'^(\w+)\s+\1',
            r'^(Все|Больше|Подробнее|Далее|Назад|Вперед)',
            r'^(Образование|Карьера|Партнёрства|Поддержка|Услуги|Продукты|Тарифы)\s*$',
        ]
        
        for pattern in unwanted_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        
        return False

    def _format_line(self, line: str) -> str:
        if not line.endswith(('.', '!', '?', ':', ';')):
            line = line + '.'
        
        if line and line[0].islower():
            line = line[0].upper() + line[1:]
        
        return line

    def _post_process_content(self, content_lines: List[str]) -> str:
        if not content_lines:
            return ""
        
        result = '\n'.join(content_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            if line.strip() and len(line.split()) >= 4:
                filtered_lines.append(line)
            elif line.startswith('Тема:'):
                filtered_lines.append(line)
        
        result = '\n'.join(filtered_lines)
        result = result.strip()
        
        return result

    def _is_section_header(self, line: str) -> bool:
        if line.endswith(':') and len(line.split()) <= 4:
            banking_keywords = [
                'услуги', 'продукты', 'условия', 'тарифы', 'карты', 'кредиты',
                'депозиты', 'инвестиции', 'страхование', 'образование', 'карьера',
                'магистратура', 'стипендии', 'гранты', 'мероприятия', 'партнёрства',
                'сервисы', 'курсы', 'информация'
            ]
            
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in banking_keywords):
                return True
        
        return False

    def _is_navigation_line(self, line: str) -> bool:
        nav_patterns = [
            r'^(Главная|Меню|Больше|Подробнее|Список|Выберите)',
            r'Карта сайта$',
            r'Поддержка$', 
            r'Стать партнёром$',
            r'Перейти к содержимому',
        ]
        
        for pattern in nav_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False

    def _is_technical_line(self, line: str) -> bool:
        if re.search(r'(USD|EUR|CNY).*\d+', line):
            return True
        
        if re.match(r'^[\s\|\-\=\+\>\<]+$', line):
            return True
        
        if re.search(r'\d+[,\.]\d+\s*[КМГ]б', line):
            return True
        
        return False
    
    def parse_train(self) -> List[Document]:
        df_websites = pd.read_csv(self.websites_path)
        
        df_websites['cleaned_text'] = df_websites.apply(
            lambda row: self._clean_page_content(row['title'], row['text']), 
            axis=1
        )
        
        df_websites = df_websites[df_websites['cleaned_text'].str.len() > 50]
        
        documents = []
        for idx, row in df_websites.iterrows():
            doc = Document(
                page_content=row['cleaned_text'],
                metadata={
                    'web_id': row['web_id'], 
                    'title': row['title'],
                    'url': row['url']
                }
            )
            documents.append(doc)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100,
            separators=['\n\n', '\n', '.', '!', '?', ';', ':', ',', ' ']
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    
    def parse_questions(self) -> List[List]:
        df_questions = pd.read_csv(self.questions_path)
        questions_list = [[q_id, query] for q_id, query in zip(df_questions['q_id'], df_questions['query'])]
        return questions_list
