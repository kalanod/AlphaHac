import re
import string

import requests

try:
    import pymorphy3

    PYMYORPHY_AVAILABLE = True
except ImportError:
    PYMYORPHY_AVAILABLE = False


class LLMAdapter:
    from openai import OpenAI

    def __init__(self, use_lemmatization=True):
        self.use_lemmatization = use_lemmatization and PYMYORPHY_AVAILABLE

        if self.use_lemmatization:
            self.morph = pymorphy3.MorphAnalyzer()

        self._init_word_lists()

    def _init_word_lists(self):
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а',
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
            'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
            'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него',
            'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом',
            'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо',
            'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без',
            'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда',
            'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним',
            'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец',
            'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через',
            'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три',
            'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда',
            'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда',
            'конечно', 'всю', 'между'
        }

        self.question_words = {
            'что', 'как', 'почему', 'зачем', 'когда', 'где', 'куда', 'откуда',
            'кто', 'чей', 'кого', 'кому', 'кем', 'какой', 'какая', 'какое',
            'какие', 'сколько', 'насколько'
        }

    def normalize(self, question):

        if not question or not isinstance(question, str):
            return ""

        normalized = self._basic_clean(question)
        normalized = self._remove_punctuation(normalized)
        if self.use_lemmatization:
            normalized = self._lemmatize_text(normalized)
        normalized = self._remove_question_words(normalized)
        normalized = self._remove_stop_words(normalized)

        return normalized

    def _basic_clean(self, text):
        cleaned = text.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.lower()
        return cleaned

    def _remove_punctuation(self, text):
        punctuation = string.punctuation.replace('-', '')
        return text.translate(str.maketrans('', '', punctuation))

    def _lemmatize_text(self, text):
        if not self.use_lemmatization:
            return text

        words = text.split()
        lemmas = []

        for word in words:
            if word in self.stop_words or len(word) < 2:
                lemmas.append(word)
                continue
            parsed = self.morph.parse(word)[0]
            lemma = parsed.normal_form
            lemmas.append(lemma)

        return ' '.join(lemmas)

    def _remove_question_words(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.question_words]
        return ' '.join(filtered_words)

    def _remove_stop_words(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def magicWand(self, documents):
        # тут волшебство
        for _, document in enumerate(documents):
            prompt = f"""
            Преобразуй текст WEB страницы в полезное содержимое для RAG.
Удали всё незначимое: меню, кнопки, навигацию, рекламу, служебные элементы, SEO-текст, повторяющиеся фразы, вводные слова и общие формулировки.
Оставь только конкретные факты, функции, характеристики, описания услуг, условий, процессов или правил, которые реально могут пригодиться для ответа на вопросы.
Переформулируй длинные фрагменты в чёткие, информативные предложения.
Исключи шум, воду, эмоциональные вставки и маркетинг.
Выводи только очищенный информативный текст, без добавлений и рассуждений.

Заголовок WEB страницы:
{document.metadata.get('title')}

WEB страница:
{document.page_content}"""
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": "Bearer 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",
                "HTTP-Referer": "http://localhost",
                "X-Title": "My App",
            }
            data = {
                "model": "openrouter/sherlock-dash-alpha",  # выбери любую модель
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            r = requests.post(url, json=data, headers=headers)
            r.raise_for_status()
            documents[_].page_content =  r.json()["choices"][0]["message"]["content"]

        return documents
