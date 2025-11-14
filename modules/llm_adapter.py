import re
import string
#from typing import Optional

try:
    import pymorphy3
    PYMYORPHY_AVAILABLE = True
except ImportError:
    PYMYORPHY_AVAILABLE = False

class LLMAdapter:
    def __init__(self, use_lemmatization = True):
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
