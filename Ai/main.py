#  ЧТО ТУТ ПРОИСХОДИТ (коротко):
#  1. Загружаем ~5000 русских анекдотов
#  2. Разбиваем текст на символы (каждый символ = токен)
#  3. Обучаем Трансформер (тот самый «T» в GPT) предсказывать
#     следующий символ по предыдущим
#  4. После обучения генерируем новые шутки по теме
#  5. Пользователь оценивает шутки — ИИ учится вашим предпочтениям
# =============================================================================

# ─── Импорты ──────────────────────────────────────────────────────────────────
# Встроенные модули Python (не нужно устанавливать)
import csv      # для работы с CSV-файлами (чтение/запись оценок)
import math     # математические функции (например, sqrt для внимания)
import os       # работа с файловой системой (создание папок, проверка файлов)
import random   # генератор случайных чисел (используется не напрямую, но нужен)
import re       # регулярные выражения (разбиваем текст на слова)
from collections import defaultdict  # словарь с умолчательным значением

# Внешние библиотеки (нужно устанавливать через pip)
import torch             # PyTorch — главный фреймворк для нейросетей
import torch.nn as nn    # nn — модуль с готовыми слоями (Linear, Embedding и т.д.)
from datasets import load_dataset  # HuggingFace Datasets — скачивает датасеты
from torch.utils.data import DataLoader, Dataset  # инструменты для загрузки данных


from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ─── Константы ────────────────────────────────────────────────────────────────
# Константы — это переменные, которые НЕ меняются во время работы программы.
# Они вынесены наверх, чтобы их было удобно настраивать.

DATASET_NAME   = "IgorVolochay/russian_jokes"  # имя датасета с HuggingFace
RATINGS_FILE   = "ratings.csv"                  # файл, куда сохраняем оценки
MODEL_FILE     = "joke_model.pt"                # файл с весами обученной модели
TOKENIZER_FILE = "joke_tokenizer.pt"            # файл со словарём токенов
MAX_JOKES      = 50000                          # сколько шуток загружать макс.
MIN_JOKE_LENGTH = 20                            # минимальная длина шутки (символов)
MAX_JOKE_LENGTH = 500                           # максимальная длина шутки (символов)
GOOD_RATING    = 4                              # оценка 4+ считается хорошей
BAD_RATING     = 2                              # оценка 2- считается плохой

# Гиперпараметры модели — меняй если нужно
# Гиперпараметры — это настройки архитектуры нейросети, которые мы выбираем ДО обучения
DIM      = 384   # размерность эмбеддингов (чем больше, тем умнее, но медленнее)
DEPTH    = 8     # количество слоёв Трансформера (глубина сети)
HEADS    = 6     # количество «голов» внимания (сколько точек зрения одновременно)
MAX_LEN  = 200   # максимальная длина последовательности в токенах
EPOCHS   = 20    # сколько раз пройдём весь датасет
BATCH    = 64    # сколько примеров обрабатываем за один шаг

# Создаём папку Ai, если её нет
# os.makedirs("Ai", exist_ok=True)

# Если файла с оценками нет — создаём с заголовком
if not os.path.exists(RATINGS_FILE):
    with open(RATINGS_FILE, "w", encoding="utf-8") as f:
        f.write("topic,joke,rating\n")  # заголовок CSV

# ─── Загрузка датасета ────────────────────────────────────────────────────────
# Датасет — это набор данных, на котором мы учим нейросеть.
# В нашем случае — коллекция русских анекдотов.

def split_words(text):
    """Разбивает текст на слова — пригодится для обучения по оценкам."""
    text = text.lower()  # приводим к нижнему регистру (чтобы «Шутка» = «шутка»)
    # Ищем все последовательности русских/английских букв и цифр
    return re.findall(r"[а-яёa-z0-9]+", text)


def find_text_column(dataset_part):
    """Ищет название колонки с текстом шуток (может называться по-разному)."""
    # Список возможных названий колонки с текстом
    possible_names = ["jokes", "joke", "text", "content", "anekdot", "answer"]
    for name in possible_names:
        if name in dataset_part.column_names:
            return name  # нашли подходящее название
    # Если ни одно не подошло — ищем любую строковую колонку
    for name in dataset_part.column_names:
        if isinstance(dataset_part[0][name], str):
            return name
    # Если ничего не нашли — ошибка
    raise ValueError("В датасете не найдена текстовая колонка со шутками.")


def load_jokes():
    print("Загружаю датасет с русскими шутками...")

    dataset = load_dataset(DATASET_NAME)
    split_name = "train" if "train" in dataset else list(dataset.keys())[0]
    dataset_part = dataset[split_name]

    text_column = find_text_column(dataset_part)

    jokes = []

    for row in dataset_part:
        joke = re.sub(r"\s+", " ", str(row[text_column]).strip())

        if MIN_JOKE_LENGTH <= len(joke) <= MAX_JOKE_LENGTH:
            jokes.append(joke)

    # 🔥 ВАЖНО: перемешиваем, а не берём первые
    random.shuffle(jokes)

    # ограничение уже после shuffle
    jokes = jokes[:MAX_JOKES]

    print(f"Загружено шуток: {len(jokes)}")
    return jokes

# ─── Токенизатор ──────────────────────────────────────────────────────────────
# Токенизатор превращает текст в числа (токены), потому что нейросети
# умеют работать ТОЛЬКО с числами, не с буквами.
# Мы используем символьную токенизацию — каждый символ = один токен.
# Это самый простой способ. ChatGPT использует более сложный (BPE),
# но принцип тот же.

# class CharTokenizer:
#     """Символьный токенизатор — каждый символ = один токен.
    
#     Принцип: составляем словарь всех символов, которые встречаются в тексте,
#     и каждому даём номер. Теперь текст можно представить как список чисел.
#     """

#     def __init__(self, texts):
#         # texts — список всех шуток. Собираем ВСЕ уникальные символы из них
#         all_chars = sorted(set("".join(texts)))
        
#         # Словарь: 4 специальных токена + все найденные символы
#         self.vocab  = ["<pad>", "<unk>", "<s>", "</s>"] + all_chars
#         # <pad>  — заполнитель (padding), чтобы все последовательности были одной длины
#         # <unk>  — неизвестный символ (если встретится новый на генерации)
#         # <s>    — начало последовательности (start)
#         # </s>   — конец последовательности (end)
        
#         # Создаём два словаря: символ → номер и номер → символ
#         self.ch2id  = {ch: i for i, ch in enumerate(self.vocab)}  # 'а' → 5
#         self.id2ch  = {i: ch for i, ch in enumerate(self.vocab)}  # 5 → 'а'
        
#         self.vocab_size = len(self.vocab)  # размер словаря (сколько всего токенов)
#         self.pad_id = 0      # номер токена <pad>
#         self.eos_id = self.ch2id["</s>"]  # номер токена конца строки

#     def encode(self, text):
#         """Превращает строку в список чисел (токенов).
        
#         Пример: encode("привет") → [4, 27, 16, 12, 9, 28]
#         """
#         # Для каждого символа берём его номер из словаря.
#         # Если символа нет в словаре — используем <unk> (номер 1)
#         return [self.ch2id.get(ch, 1) for ch in text]

#     def decode(self, ids):
#         """Превращает список чисел обратно в строку.
        
#         Пример: decode([4, 27, 16, 12, 9, 28]) → "привет"
#         """
#         return "".join(self.id2ch.get(i, "?") for i in ids)

#     def save(self, path):
#         """Сохраняет токенизатор в файл (словарь символов)."""
#         torch.save({"vocab": self.vocab}, path)

#     @classmethod
#     def load(cls, path):
#         """Загружает токенизатор из файла.
        
#         @classmethod — это метод класса, а не экземпляра.
#         Он вызывается как CharTokenizer.load(path) и возвращает новый объект.
#         """
#         obj = cls.__new__(cls)  # создаём объект без вызова __init__
#         data = torch.load(path, weights_only=True)  # загружаем словарь
#         obj.vocab     = data["vocab"]
#         obj.ch2id     = {ch: i for i, ch in enumerate(obj.vocab)}
#         obj.id2ch     = {i: ch for i, ch in enumerate(obj.vocab)}
#         obj.vocab_size = len(obj.vocab)
#         obj.pad_id    = 0
#         obj.eos_id    = obj.ch2id["</s>"]
#         return obj

class BPETokenizer:
    def __init__(self, texts=None, path=None):
        if path:
            self.tokenizer = Tokenizer.from_file(path)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))
            self.tokenizer.pre_tokenizer = Whitespace()

            trainer = BpeTrainer(
                vocab_size=8000,   # 🔥 важный параметр
                special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
            )

            self.tokenizer.train_from_iterator(texts, trainer)

        self.pad_id = 0
        self.eos_id = self.tokenizer.token_to_id("</s>")

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path):
        return cls(path=path)

# ─── Датасет ──────────────────────────────────────────────────────────────────
# Датасет в PyTorch — это класс, который выдаёт пары (вход, цель).
# Мы учим модель предсказывать следующий символ, поэтому:
#   вход  = "<s>шутка тек"
#   цель  = "шутка текст</
# То есть цель — это вход, сдвинутый на 1 символ вправо.

class JokeDataset(Dataset):
    def __init__(self, jokes, tokenizer, max_len=MAX_LEN):
        self.samples = []  # список примеров для обучения
        for joke in jokes:
            # Превращаем шутку в числа и добавляем маркеры начала/конца
            ids = tokenizer.encode("<s>" + joke + "</s>")
            # Режем на кусочки по max_len символов
            # Шутка может быть длиннее max_len, поэтому нарезаем с перекрытием
            for i in range(0, len(ids) - 1, max_len):
                chunk = ids[i: i + max_len + 1]  # берём кусочек
                if len(chunk) > 10:  # слишком короткие кусочки отбрасываем
                    self.samples.append(chunk)

    def __len__(self):
        """Сколько всего примеров в датасете."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Возвращает один пример (вход и цель) по индексу.
        
        ВАЖНО: вход — это все символы КРОМЕ последнего,
               цель  — это все символы КРОМЕ первого.
        То есть модель учится предсказывать «следующий символ».
        """
        chunk = self.samples[idx]
        # Вход: всё кроме последнего символа
        # Цель: всё кроме первого символа (сдвиг на 1)
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


def pad_collate(batch):
    """Собирает несколько примеров в один батч и дополняет до одинаковой длины.
    
    Проблема: разные примеры разной длины. Нейросети нужно, чтобы
    все примеры в батче были одинаковой длины. Решение: дополняем
    короткие примеры токенами <pad> (номер 0).
    """
    xs, ys = zip(*batch)  # разделяем на входы и цели
    max_l = max(x.size(0) for x in xs)  # находим самую длинную последовательность
    
    # Для каждого примера: если он короче max_l, дополняем нулями справа
    # nn.functional.pad(x, (0, max_l - x.size(0))) добавляет нули в конец
    xs = torch.stack([nn.functional.pad(x, (0, max_l - x.size(0))) for x in xs])
    ys = torch.stack([nn.functional.pad(y, (0, max_l - y.size(0))) for y in ys])
    return xs, ys

# ─── Модель (маленький GPT) ───────────────────────────────────────────────────
#    ╔══════════════════════════════════════════════════════════╗
#    ║  ЧТО ТАКОЕ GPT (очень кратко)                          ║
#    ║                                                         ║
#    ║  GPT = Generative Pre-trained Transformer               ║
#    ║  Это нейросеть, которая предсказывает следующий         ║
#    ║  токен, глядя на все предыдущие. «Внимание» (Attention) ║
#    ║  позволяет каждому токену «заглядывать» на все          ║
#    ║  предыдущие и решать, какие из них важны.               ║
#    ╚══════════════════════════════════════════════════════════╝

class SelfAttention(nn.Module):
    """Один слой «самовнимания» (Self-Attention).
    
    Идея: пусть у нас есть предложение «Кот сел на ковёр и ...».
    Чтобы предсказать следующее слово, нужно понять, что «кот» — это
    подлежащее, а «сел» — сказуемое. Внимание вычисляет, какие слова
    связаны друг с другом, и насколько сильно.
    
    Механизм: для каждого слова мы создаём 3 вектора:
      - Q (Query) — «запрос»: что я ищу?
      - K (Key) — «ключ»: что я предлагаю?
      - V (Value) — «значение»: что я даю, если меня выберут?
    
    Слово «смотрит» на все предыдущие: его Q сравнивается с их K,
    получается «вес внимания» (на кого смотреть сильнее).
    Потом эти веса умножаются на V всех слов, и результаты складываются.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.heads    = heads     # количество голов внимания
        self.head_dim = dim // heads  # размерность одной головы
        # Один большой слой, который считает Q, K и V сразу
        # dim * 3 — потому что Q, K и V имеют размерность dim
        self.qkv = nn.Linear(dim, dim * 3)  # Query, Key, Value в одной матрице
        self.out = nn.Linear(dim, dim)      # выходной линейный слой

    def forward(self, x):
        # x — входной тензор формы (Batch, Time, Channels)
        # B = количество примеров в батче
        # T = длина последовательности (сколько токенов)
        # C = размерность эмбеддинга (dim)
        B, T, C = x.shape
        
        # Считаем Q, K, V одним умножением на матрицу, потом разделяем
        # qkv(x) даёт тензор (B, T, C*3), chunk делит на 3 части по оси C
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # Разделяем на головы: (B, T, heads, head_dim)
        # transpos(1,2) меняет T и heads местами: (B, heads, T, head_dim)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Считаем внимание:
        # 1. Q * K^T (матричное умножение) — получаем «оценки внимания»
        # 2. Делим на sqrt(head_dim) — это масштабирование, чтобы градиенты не взрывались
        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale  # (B, heads, T, T)
        
        # Маска — запрещаем токенам «заглядывать в будущее»
        # Это ключевая идея decoder-only трансформеров (как GPT):
        # токен на позиции i может смотреть только на позиции j <= i
        # Иначе модель будет «жульничать» — подглядывать ответ
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # Заменяем верхний треугольник (будущее) на -∞
        # После softmax это даст 0 — то есть запрет на внимание в будущее
        scores = scores.masked_fill(mask, float("-inf"))
        
        # Softmax превращает оценки в вероятности (сумма = 1)
        attn = torch.softmax(scores, dim=-1)  # (B, heads, T, T)
        
        # Умножаем вероятности на V — получаем взвешенную сумму
        # transpose(1,2) и reshape возвращают форму (B, T, C)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        
        # Выходной линейный слой
        return self.out(out)


class TransformerBlock(nn.Module):
    """Один блок Трансформера.
    
    Структура:
       X → LayerNorm → SelfAttention → + (residual) →
        → LayerNorm → FeedForward → + (residual)
    
    Residual connection (остаточная связь): x = x + f(x).
    Зачем? Чтобы градиент мог «обойти» слой напрямую — это помогает
    обучению глубоких сетей (решает проблему затухающих градиентов).
    
    LayerNorm — нормализация слоя: приводит значения к единому масштабу.
    Это ускоряет и стабилизирует обучение.
    
    FeedForward — простая нейросеть: Linear → GELU → Linear.
    GELU — функция активации (чуть более плавная версия ReLU).
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.attn = SelfAttention(dim, heads)  # слой внимания
        self.ff   = nn.Sequential(              # полносвязная сеть (FFN)
            nn.Linear(dim, dim * 4),   # расширяем размерность в 4 раза
            nn.GELU(),                  # нелинейная активация
            nn.Linear(dim * 4, dim),   # сжимаем обратно
        )
        self.ln1 = nn.LayerNorm(dim)  # нормализация перед вниманием
        self.ln2 = nn.LayerNorm(dim)  # нормализация перед FFN

    def forward(self, x):
        # Внимание с остаточной связью:
        x = x + self.attn(self.ln1(x))
        # FeedForward с остаточной связью:
        x = x + self.ff(self.ln2(x))
        return x


class TinyJokeGPT(nn.Module):
    """Наша маленькая GPT-модель.
    
    Схема:
      Токены (числа) → Embedding (векторы) + Positional Embedding (позиции) →
      → N блоков Трансформера → LayerNorm → Linear → предсказание следующего токена
    
    Embedding — это таблица, где каждому токену соответствует вектор.
    Вектор — это «смысл» токена в многомерном пространстве.
    
    Positional Embedding — говорит модели, на каком месте стоит токен.
    Без этого все токены были бы «перепутаны» — внимание не знает порядок слов.
    """

    def __init__(self, vocab_size, dim=DIM, depth=DEPTH, heads=HEADS, max_len=MAX_LEN):
        super().__init__()
        # Таблица эмбеддингов токенов: vocab_size строк, dim столбцов
        self.token_emb = nn.Embedding(vocab_size, dim)
        # Таблица позиционных эмбеддингов: max_len строк, dim столбцов
        self.pos_emb   = nn.Embedding(max_len, dim)
        
        # Стек из depth блоков Трансформера
        self.blocks    = nn.Sequential(
            *[TransformerBlock(dim, heads) for _ in range(depth)]
        )
        
        self.ln_final  = nn.LayerNorm(dim)    # финальная нормализация
        self.head      = nn.Linear(dim, vocab_size)  # предсказание следующего токена

    def forward(self, x):
        B, T = x.shape  # B = размер батча, T = длина последовательности
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # позиции [0, 1, 2, ..., T-1]
        
        # Складываем эмбеддинги токенов и позиций
        # Вектор каждого токена = смысл токена + его позиция
        x = self.token_emb(x) + self.pos_emb(pos)
        
        # Пропускаем через все блоки Трансформера
        x = self.blocks(x)
        
        # Финальный линейный слой: из вектора размерности dim
        # получаем вероятности для каждого токена из словаря
        return self.head(self.ln_final(x))

# ─── Обучение и генерация ─────────────────────────────────────────────────────
# Обучение — это процесс подбора весов нейросети так, чтобы
# её предсказания совпадали с правильными ответами.

def train_model(model, dataset, tokenizer, epochs=EPOCHS):
    """Обучает модель предсказывать следующий символ.
    
    Как это работает:
      1. Берём последовательность символов (вход)
      2. Модель предсказывает следующий символ (выход)
      3. Сравниваем предсказание с реальным следующим символом (ошибка)
      4. Считаем градиенты — в какую сторону менять веса, чтобы ошибка уменьшилась
      5. Делаем шаг оптимизатором — чуть-чуть меняем веса
      6. Повторяем миллионы раз
    """
    # DataLoader — разбивает датасет на батчи и перемешивает
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=pad_collate)
    
    # AdamW — оптимизатор, который решает, КАК менять веса
    # lr (learning rate) — скорость обучения: как сильно менять веса за шаг
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Функция потерь — считает, насколько ошиблась модель
    # CrossEntropyLoss — для задачи классификации (какой токен следующий?)
    # ignore_index=0 — игнорируем токены <pad> (не учимся на пустоте)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)

    # Проверяем, есть ли сохранённый прогресс
    # Если обучение прервалось, можем продолжить с того же места
    start_epoch = 0
    if os.path.exists("joke_checkpoint.pt"):
        checkpoint = torch.load("joke_checkpoint.pt", weights_only=True)
        model.load_state_dict(checkpoint["model"])          # загружаем веса модели
        optimizer.load_state_dict(checkpoint["optimizer"])  # загружаем состояние оптимизатора
        start_epoch = checkpoint["epoch"]                   # с какой эпохи продолжаем
        print(f"Продолжаю с эпохи {start_epoch + 1}")

    model.train()  # переключаем модель в режим обучения (включаем dropout и т.д.)
    
    for epoch in range(start_epoch, epochs):
        total = 0  # сумма потерь за эпоху
        
        for x, y in loader:
            # x — входные токены (B, T)
            # y — целевые токены (B, T) — сдвинуты на 1 вправо
            
            logits = model(x)  # предсказание модели (B, T, vocab_size)
            
            # Сравниваем предсказание с правильным ответом
            # view(-1, ...) — превращаем в плоский тензор для подсчёта loss
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()  # обнуляем градиенты (они накапливаются)
            loss.backward()        # считаем градиенты (обратное распространение)
            
            # Обрезаем градиенты — чтобы они не стали слишком большими
            # (проблема «взрывающихся градиентов»)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()  # делаем шаг оптимизации (меняем веса)
            total += loss.item()  # добавляем потерю к общей сумме

        avg_loss = total / len(loader)  # средняя потеря за эпоху
        print(f"  Эпоха {epoch + 1}/{epochs} | loss: {avg_loss:.4f}")

        # Сохраняем прогресс после каждой эпохи
        torch.save({
            "epoch":     epoch + 1,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, "joke_checkpoint.pt")

    # Финальное сохранение готовой модели
    torch.save(model.state_dict(), MODEL_FILE)
    tokenizer.save(TOKENIZER_FILE)
    os.remove("joke_checkpoint.pt")  # промежуточный чекпоинт больше не нужен


def generate_joke(model, tokenizer, prompt, max_new=150, temperature=0.8):
    """Генерирует текст, предсказывая по одному токену за раз.
    
    Как работает генерация:
      1. Начинаем с промпта (темы)
      2. Модель предсказывает следующий токен
      3. Добавляем его к последовательности
      4. Повторяем, пока не встретим </s> или не превысим лимит
    
    Temperature — «температура» творчества:
      - 0.0 — всегда выбирает самый вероятный токен (скучно, но безопасно)
      - 1.0 — выбирает случайно по вероятностям (креативно, но может чушь)
      - >1.0 — почти случайный выбор (хаос)
    """
    model.eval()  # переключаем в режим оценки (выключаем dropout)
    
    # Превращаем промпт в токены и добавляем размерность батча
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():  # не считаем градиенты (экономим память)
        for _ in range(max_new):
            # Обрезаем до MAX_LEN если последовательность слишком длинная
            inp = ids[:, -MAX_LEN:]
            
            logits = model(inp)  # предсказание (B, T, vocab_size)
            
            # Берём только последний токен (logits[0, -1])
            # Делим на temperature — чем выше T, тем равномернее распределение
            next_logits = logits[0, -1] / temperature
            
            # Превращаем логиты в вероятности через softmax
            probs = torch.softmax(next_logits, dim=-1)
            
            # Сэмплируем (выбираем) следующий токен согласно вероятностям
            # torch.multinomial — случайный выбор с заданными вероятностями
            topk = 40
            values, indices = torch.topk(probs, topk)
            values = values / values.sum()
            next_id = indices[torch.multinomial(values, 1)].item()

            # Если модель решила закончить — выходим
            if next_id == tokenizer.eos_id:
                break

            # Добавляем новый токен к последовательности
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    # Превращаем токены обратно в текст
    result = tokenizer.decode(ids[0].tolist())
    # Убираем служебные токены из результата
    result = result.replace("<s>", "").replace("</s>", "").strip()
    return result

# ─── Главный класс ИИ ─────────────────────────────────────────────────────────
# JokeAI — это «интеллект» целиком: у него есть модель, токенизатор,
# память об оценках и способность учиться на feedback.

class JokeAI:
    def __init__(self, jokes):
        self.jokes       = jokes             # все шутки из датасета
        self.word_weights = defaultdict(float)  # веса слов (учится на оценках)
        self.used_jokes  = set()             # множество уже показанных шуток

        # Загружаем или обучаем модель
        if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
            print("Загружаю сохранённую модель...")
            self.tokenizer = BPETokenizer.load(TOKENIZER_FILE)  # загружаем токенизатор
            self.gpt = TinyJokeGPT(vocab_size=self.tokenizer.vocab_size)
            self.gpt.load_state_dict(torch.load(MODEL_FILE, weights_only=True))  # веса
            print("Модель загружена!")
        else:
            print("Обучаю свою модель с нуля (первый запуск)...")
            self.tokenizer = BPETokenizer(jokes)           # создаём токенизатор
            dataset        = JokeDataset(jokes, self.tokenizer)  # создаём датасет
            self.gpt       = TinyJokeGPT(vocab_size=self.tokenizer.vocab_size)
            # Считаем и выводим количество параметров модели
            print(f"Параметров в модели: {sum(p.numel() for p in self.gpt.parameters()):,}")
            train_model(self.gpt, dataset, self.tokenizer)
            print("Модель сохранена!")

        self.load_ratings()  # загружаем предыдущие оценки

    def generate(self, topic):
        """Генерирует шутку на заданную тему."""
        # Даём модели тему как начало шутки
        prompt = f"<s>тема: {topic}\nшутка:"
        joke   = generate_joke(self.gpt, self.tokenizer, prompt)
        return joke if joke else "Не смог придумать шутку, попробуй другую тему."

    def learn_from_rating(self, joke, rating):
        """Учится на оценке пользователя.
        
        Не переобучает нейросеть (это долго), а просто запоминает:
        какие слова в шутках понравились пользователю, а какие — нет.
        """
        words = split_words(joke)  # разбиваем шутку на слова
        
        if rating >= GOOD_RATING:    # 4 или 5 — понравилось
            change = 0.15
        elif rating <= BAD_RATING:   # 1 или 2 — не понравилось
            change = -0.15
        else:                        # 3 — нейтрально
            change = 0.02
        
        # Увеличиваем или уменьшаем вес каждого слова в шутке
        for word in words:
            self.word_weights[word] += change

    def save_rating(self, topic, joke, rating):
        """Сохраняет оценку в CSV-файл."""
        file_exists = os.path.exists(RATINGS_FILE)
        with open(RATINGS_FILE, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["topic", "joke", "rating"])  # заголовок
            writer.writerow([topic, joke, rating])  # данные

    def load_ratings(self):
        """Загружает все предыдущие оценки из CSV."""
        if not os.path.exists(RATINGS_FILE):
            return
        with open(RATINGS_FILE, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Применяем каждую оценку к весам слов
                self.learn_from_rating(row["joke"], int(row["rating"]))

# ─── Интерфейс ────────────────────────────────────────────────────────────────
# Простой диалог в консоли: пользователь пишет тему, ИИ генерирует шутку,
# пользователь оценивает, ИИ запоминает.

def ask_rating():
    """Спрашивает оценку у пользователя, проверяет ввод."""
    while True:
        text = input("Оцени шутку от 1 до 5: ").strip()
        if text in ["выход", "exit", "quit"]:
            return None  # пользователь хочет выйти
        if text in ["1", "2", "3", "4", "5"]:
            return int(text)
        print("Нужно ввести число от 1 до 5.")


def main():
    """Главная функция: запускает цикл общения с ИИ."""
    jokes = load_jokes()  # загружаем датасет шуток
    ai    = JokeAI(jokes)  # создаём или загружаем модель
    print("\nИИ готов. Пиши тему шутки или 'выход'.")

    while True:
        topic = input("\nТема шутки: ").strip().lower()
        if topic in ["выход", "exit", "quit"]:
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")
            break

        joke = ai.generate(topic)  # генерируем шутку
        print("\n" + joke)

        rating = ask_rating()  # спрашиваем оценку
        if rating is None:
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")
            break

        ai.learn_from_rating(joke, rating)  # учимся на оценке
        ai.save_rating(topic, joke, rating)  # сохраняем оценку
        print("Оценка сохранена. Следующая шутка будет учитывать твой вкус.")


# Это стандартная конструкция Python: код выполняется,
# только если файл запускают напрямую, а не импортируют
if __name__ == "__main__":
    main()
