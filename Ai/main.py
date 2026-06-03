import csv  # Подключаем csv, чтобы сохранять оценки пользователя в файл.
import os  # Подключаем os, чтобы удобно строить пути к файлам.
import random  # Подключаем random, чтобы выбирать разные шутки.
import re  # Подключаем re, чтобы разбивать текст на слова.
from collections import defaultdict  # Подключаем defaultdict, чтобы удобно хранить веса слов.

from datasets import load_dataset  # Подключаем load_dataset, чтобы загрузить датасет с Hugging Face.


DATASET_NAME = "IgorVolochay/russian_jokes"  # Название датасета с русскими шутками на Hugging Face.
RATINGS_FILE = os.path.join("ratings.csv")  # Файл, куда программа будет сохранять оценки пользователя.
MAX_JOKES = 5000  # Ограничиваем количество шуток, чтобы программа работала быстро даже на слабом компьютере.
MIN_JOKE_LENGTH = 20  # Слишком короткие тексты пропускаем, потому что они часто бесполезны.
MAX_JOKE_LENGTH = 500  # Слишком длинные истории пропускаем, чтобы программа показывала именно короткие шутки.
GOOD_RATING = 4  # Оценки 4 и 5 считаем хорошими.
BAD_RATING = 2  # Оценки 1 и 2 считаем плохими.

os.makedirs("Ai", exist_ok=True)
if not os.path.exists(RATINGS_FILE):
    with open(RATINGS_FILE, "w", encoding="utf-8") as f:
        f.write("topic,joke,rating\n")


def split_words(text):  # Создаем функцию, которая превращает текст в список слов.
    text = text.lower()  # Переводим текст в нижний регистр, чтобы "Кот" и "кот" считались одним словом.
    return re.findall(r"[а-яёa-z0-9]+", text)  # Возвращаем только слова и цифры без знаков препинания.


def find_text_column(dataset_part):  # Создаем функцию, которая ищет колонку с текстом шутки.
    possible_names = ["jokes", "joke", "text", "content", "anekdot", "answer"]  # Перечисляем самые вероятные названия колонок.
    for name in possible_names:  # Перебираем возможные названия по очереди.
        if name in dataset_part.column_names:  # Проверяем, есть ли такая колонка в датасете.
            return name  # Возвращаем найденное название колонки.
    for name in dataset_part.column_names:  # Если стандартные названия не подошли, перебираем все колонки.
        first_value = dataset_part[0][name]  # Берем первое значение из колонки.
        if isinstance(first_value, str):  # Проверяем, что значение является строкой.
            return name  # Возвращаем первую текстовую колонку.
    raise ValueError("В датасете не найдена текстовая колонка со шутками.")  # Сообщаем об ошибке, если текста нет.


def load_jokes():  # Создаем функцию загрузки шуток из датасета.
    print("Загружаю датасет с русскими шутками...")  # Показываем пользователю, что сейчас идет загрузка.
    try:  # Пробуем обычную загрузку датасета.
        dataset = load_dataset(DATASET_NAME)  # Загружаем датасет с Hugging Face по имени.
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]  # Берем train, а если его нет, берем первый раздел.
        dataset_part = dataset[split_name]  # Достаем нужный раздел датасета.
    except Exception:  # Если обычная загрузка упала из-за схемы датасета, используем CSV-файл.
        csv_path = f"hf://datasets/{DATASET_NAME}/dataset.csv"  # Создаем путь к CSV-файлу внутри репозитория Hugging Face.
        dataset_part = load_dataset("csv", data_files=csv_path, split="train")  # Загружаем тот же датасет напрямую из CSV.
    text_column = find_text_column(dataset_part)  # Находим колонку, где лежит текст шутки.
    jokes = []  # Создаем пустой список для очищенных шуток.

    for row in dataset_part:  # Проходим по строкам датасета.
        joke = str(row[text_column]).strip()  # Берем текст шутки и убираем пробелы по краям.
        joke = re.sub(r"\s+", " ", joke)  # Заменяем переносы строк и лишние пробелы одним пробелом.
        if MIN_JOKE_LENGTH <= len(joke) <= MAX_JOKE_LENGTH:  # Проверяем, что шутка не слишком короткая и не слишком длинная.
            jokes.append(joke)  # Добавляем шутку в список.
        if len(jokes) >= MAX_JOKES:  # Проверяем, набрали ли мы достаточно шуток.
            break  # Останавливаем загрузку, чтобы не держать слишком много данных.

    if not jokes:  # Проверяем, что список шуток не пустой.
        raise ValueError("Не получилось загрузить шутки из датасета.")  # Сообщаем понятную ошибку.

    print(f"Загружено шуток: {len(jokes)}")  # Показываем, сколько шуток удалось взять.
    return jokes  # Возвращаем список шуток.


class JokeAI:  # Создаем простой класс ИИ для выбора шуток.
    def __init__(self, jokes):  # Описываем создание ИИ.
        self.jokes = jokes  # Сохраняем все загруженные шутки внутри объекта.
        self.word_weights = defaultdict(float)  # Создаем веса слов, которые будут меняться от оценок.
        self.used_jokes = set()  # Создаем набор уже показанных шуток, чтобы меньше повторяться.
        self.load_ratings()  # Загружаем старые оценки и дообучаемся на них.

    def score_joke(self, joke, topic_words):  # Создаем функцию оценки шутки для выбора.
        words = split_words(joke)  # Разбиваем шутку на слова.
        topic_score = sum(3.0 for word in words if word in topic_words)  # Даем плюс за совпадение слов с темой.
        user_score = sum(self.word_weights[word] for word in words)  # Добавляем опыт прошлых оценок пользователя.
        random_score = random.uniform(0.0, 2.0)  # Добавляем немного случайности, чтобы ответы не были одинаковыми.
        repeat_penalty = -100.0 if joke in self.used_jokes else 0.0  # Сильно штрафуем шутки, которые уже показывали.
        return topic_score + user_score + random_score + repeat_penalty  # Возвращаем итоговый балл шутки.

    def generate(self, topic):  # Создаем функцию генерации, точнее выбора шутки из датасета.
        topic_words = set(split_words(topic))  # Превращаем тему пользователя в набор слов.
        sample_size = min(400, len(self.jokes))  # Ограничиваем количество кандидатов, чтобы выбор был быстрым.
        candidates = random.sample(self.jokes, sample_size)  # Берем случайные шутки-кандидаты из датасета.
        best_joke = max(candidates, key=lambda joke: self.score_joke(joke, topic_words))  # Выбираем шутку с лучшим баллом.
        self.used_jokes.add(best_joke)  # Запоминаем, что эту шутку уже показывали.
        return best_joke  # Возвращаем выбранную шутку.

    def learn_from_rating(self, joke, rating):  # Создаем функцию обучения на оценке пользователя.
        words = split_words(joke)  # Разбиваем оцененную шутку на слова.
        if rating >= GOOD_RATING:  # Проверяем, понравилась ли шутка.
            change = 0.15  # Для хорошей оценки увеличиваем веса слов.
        elif rating <= BAD_RATING:  # Проверяем, не понравилась ли шутка.
            change = -0.15  # Для плохой оценки уменьшаем веса слов.
        else:  # Обрабатываем нейтральную оценку 3.
            change = 0.02  # Для нейтральной оценки меняем веса совсем чуть-чуть.

        for word in words:  # Проходим по каждому слову из шутки.
            self.word_weights[word] += change  # Обновляем вес слова по оценке пользователя.

    def save_rating(self, topic, joke, rating):  # Создаем функцию сохранения оценки в CSV.
        file_exists = os.path.exists(RATINGS_FILE)  # Проверяем, существует ли файл с оценками.
        with open(RATINGS_FILE, "a", encoding="utf-8", newline="") as file:  # Открываем файл для добавления строки.
            writer = csv.writer(file)  # Создаем объект для записи CSV.
            if not file_exists:  # Проверяем, нужно ли записать заголовок.
                writer.writerow(["topic", "joke", "rating"])  # Записываем названия колонок.
            writer.writerow([topic, joke, rating])  # Записываем тему, шутку и оценку.

    def load_ratings(self):  # Создаем функцию загрузки старых оценок.
        if not os.path.exists(RATINGS_FILE):  # Проверяем, есть ли файл с оценками.
            return  # Если файла нет, просто выходим.
        with open(RATINGS_FILE, "r", encoding="utf-8", newline="") as file:  # Открываем файл оценок для чтения.
            reader = csv.DictReader(file)  # Читаем CSV как словарь.
            for row in reader:  # Проходим по каждой старой оценке.
                joke = row["joke"]  # Берем текст шутки.
                rating = int(row["rating"])  # Берем оценку и превращаем ее в число.
                self.learn_from_rating(joke, rating)  # Повторно обучаем ИИ на старой оценке.


def ask_rating():  # Создаем функцию, которая спрашивает оценку у пользователя.
    while True:  # Запускаем цикл, пока пользователь не введет нормальную оценку.
        text = input("Оцени шутку от 1 до 5: ").strip()  # Просим оценку и убираем лишние пробелы.
        if text in ["выход", "exit", "quit"]:  # Проверяем, хочет ли пользователь выйти.
            return None  # Возвращаем None как знак выхода.
        if text in ["1", "2", "3", "4", "5"]:  # Проверяем, что введена оценка от 1 до 5.
            return int(text)  # Возвращаем оценку числом.
        print("Нужно ввести число от 1 до 5.")  # Подсказываем правильный формат ввода.


def main():  # Создаем главную функцию программы.
    jokes = load_jokes()  # Загружаем шутки из датасета.
    ai = JokeAI(jokes)  # Создаем ИИ и обучаем его на старых оценках.
    print("ИИ готов. Пиши тему шутки или 'выход'.")  # Сообщаем, что можно начинать.

    while True:  # Запускаем основной цикл программы.
        topic = input("\nТема шутки: ").strip().lower()  # Просим тему шутки у пользователя.
        if topic in ["выход", "exit", "quit"]:  # Проверяем, хочет ли пользователь выйти.
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")  # Печатаем прощальное сообщение.
            break  # Завершаем цикл.

        joke = ai.generate(topic)  # Генерируем шутку по теме.
        print("\n" + joke)  # Печатаем шутку с пустой строкой перед ней.
        rating = ask_rating()  # Просим пользователя оценить шутку.
        if rating is None:  # Проверяем, попросил ли пользователь выйти во время оценки.
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")  # Печатаем прощальное сообщение.
            break  # Завершаем цикл.
        ai.learn_from_rating(joke, rating)  # Дообучаем ИИ на новой оценке.
        ai.save_rating(topic, joke, rating)  # Сохраняем оценку в файл.
        print("Оценка сохранена. Следующая шутка будет учитывать твой вкус.")  # Сообщаем, что обучение применилось.


if __name__ == "__main__":  # Проверяем, запустили ли файл напрямую.
    main()  # Запускаем главную функцию.
