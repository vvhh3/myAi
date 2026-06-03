import csv
import math
import os
import random
import re
from collections import defaultdict

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# ─── Константы ────────────────────────────────────────────────────────────────

DATASET_NAME   = "IgorVolochay/russian_jokes"
RATINGS_FILE   = "ratings.csv"
MODEL_FILE     = "joke_model.pt"
TOKENIZER_FILE = "joke_tokenizer.pt"
MAX_JOKES      = 5000
MIN_JOKE_LENGTH = 20
MAX_JOKE_LENGTH = 500
GOOD_RATING    = 4
BAD_RATING     = 2

# Гиперпараметры модели — меняй если нужно
DIM      = 256   # размер векторов (больше = умнее, но медленнее)
DEPTH    = 6     # количество слоёв трансформера
HEADS    = 4     # количество голов внимания
MAX_LEN  = 200   # максимальная длина в токенах
EPOCHS   = 40    # количество эпох обучения
BATCH    = 32    # размер батча

os.makedirs("Ai", exist_ok=True)
if not os.path.exists(RATINGS_FILE):
    with open(RATINGS_FILE, "w", encoding="utf-8") as f:
        f.write("topic,joke,rating\n")

# ─── Загрузка датасета ────────────────────────────────────────────────────────

def split_words(text):
    text = text.lower()
    return re.findall(r"[а-яёa-z0-9]+", text)


def find_text_column(dataset_part):
    possible_names = ["jokes", "joke", "text", "content", "anekdot", "answer"]
    for name in possible_names:
        if name in dataset_part.column_names:
            return name
    for name in dataset_part.column_names:
        if isinstance(dataset_part[0][name], str):
            return name
    raise ValueError("В датасете не найдена текстовая колонка со шутками.")


def load_jokes():
    print("Загружаю датасет с русскими шутками...")
    try:
        dataset = load_dataset(DATASET_NAME)
        split_name = "train" if "train" in dataset else list(dataset.keys())[0]
        dataset_part = dataset[split_name]
    except Exception:
        csv_path = f"hf://datasets/{DATASET_NAME}/dataset.csv"
        dataset_part = load_dataset("csv", data_files=csv_path, split="train")

    text_column = find_text_column(dataset_part)
    jokes = []
    for row in dataset_part:
        joke = re.sub(r"\s+", " ", str(row[text_column]).strip())
        if MIN_JOKE_LENGTH <= len(joke) <= MAX_JOKE_LENGTH:
            jokes.append(joke)
        if len(jokes) >= MAX_JOKES:
            break

    if not jokes:
        raise ValueError("Не получилось загрузить шутки из датасета.")
    print(f"Загружено шуток: {len(jokes)}")
    return jokes

# ─── Токенизатор ──────────────────────────────────────────────────────────────

class CharTokenizer:
    """Символьный токенизатор — каждый символ = один токен."""

    def __init__(self, texts):
        all_chars = sorted(set("".join(texts)))
        self.vocab  = ["<pad>", "<unk>", "<s>", "</s>"] + all_chars
        self.ch2id  = {ch: i for i, ch in enumerate(self.vocab)}
        self.id2ch  = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_id = 0
        self.eos_id = self.ch2id["</s>"]

    def encode(self, text):
        return [self.ch2id.get(ch, 1) for ch in text]

    def decode(self, ids):
        return "".join(self.id2ch.get(i, "?") for i in ids)

    def save(self, path):
        torch.save({"vocab": self.vocab}, path)

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        data = torch.load(path, weights_only=True)
        obj.vocab     = data["vocab"]
        obj.ch2id     = {ch: i for i, ch in enumerate(obj.vocab)}
        obj.id2ch     = {i: ch for i, ch in enumerate(obj.vocab)}
        obj.vocab_size = len(obj.vocab)
        obj.pad_id    = 0
        obj.eos_id    = obj.ch2id["</s>"]
        return obj

# ─── Датасет ──────────────────────────────────────────────────────────────────

class JokeDataset(Dataset):
    def __init__(self, jokes, tokenizer, max_len=MAX_LEN):
        self.samples = []
        for joke in jokes:
            ids = tokenizer.encode("<s>" + joke + "</s>")
            for i in range(0, len(ids) - 1, max_len):
                chunk = ids[i: i + max_len + 1]
                if len(chunk) > 10:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        return (torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:],  dtype=torch.long))


def pad_collate(batch):
    xs, ys = zip(*batch)
    max_l = max(x.size(0) for x in xs)
    xs = torch.stack([nn.functional.pad(x, (0, max_l - x.size(0))) for x in xs])
    ys = torch.stack([nn.functional.pad(y, (0, max_l - y.size(0))) for y in ys])
    return xs, ys

# ─── Модель (маленький GPT) ───────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads    = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        scale  = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-2, -1)) / scale
        mask   = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        attn   = torch.softmax(scores, dim=-1)
        out    = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = SelfAttention(dim, heads)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyJokeGPT(nn.Module):
    def __init__(self, vocab_size, dim=DIM, depth=DEPTH, heads=HEADS, max_len=MAX_LEN):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb   = nn.Embedding(max_len, dim)
        self.blocks    = nn.Sequential(*[TransformerBlock(dim, heads) for _ in range(depth)])
        self.ln_final  = nn.LayerNorm(dim)
        self.head      = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = self.token_emb(x) + self.pos_emb(pos)
        x   = self.blocks(x)
        return self.head(self.ln_final(x))

# ─── Обучение и генерация ─────────────────────────────────────────────────────

def train_model(model, dataset, epochs=EPOCHS):
    loader    = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=pad_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(epochs):
        total = 0
        for x, y in loader:
            logits = model(x)
            loss   = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        print(f"  Эпоха {epoch + 1}/{epochs} | loss: {total / len(loader):.4f}")


def generate_joke(model, tokenizer, prompt, max_new=150, temperature=0.8):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new):
            # Обрезаем до MAX_LEN если слишком длинно
            inp    = ids[:, -MAX_LEN:]
            logits = model(inp)
            next_logits = logits[0, -1] / temperature
            probs   = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == tokenizer.eos_id:
                break

            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    result = tokenizer.decode(ids[0].tolist())
    # Убираем служебные токены
    result = result.replace("<s>", "").replace("</s>", "").strip()
    return result

# ─── Главный класс ИИ ─────────────────────────────────────────────────────────

class JokeAI:
    def __init__(self, jokes):
        self.jokes       = jokes
        self.word_weights = defaultdict(float)
        self.used_jokes  = set()

        # Загружаем или обучаем модель
        if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
            print("Загружаю сохранённую модель...")
            self.tokenizer = CharTokenizer.load(TOKENIZER_FILE)
            self.gpt = TinyJokeGPT(vocab_size=self.tokenizer.vocab_size)
            self.gpt.load_state_dict(torch.load(MODEL_FILE, weights_only=True))
            print("Модель загружена!")
        else:
            print("Обучаю свою модель с нуля (первый запуск)...")
            self.tokenizer = CharTokenizer(jokes)
            dataset        = JokeDataset(jokes, self.tokenizer)
            self.gpt       = TinyJokeGPT(vocab_size=self.tokenizer.vocab_size)
            print(f"Параметров в модели: {sum(p.numel() for p in self.gpt.parameters()):,}")
            train_model(self.gpt, dataset)
            torch.save(self.gpt.state_dict(), MODEL_FILE)
            self.tokenizer.save(TOKENIZER_FILE)
            print("Модель сохранена!")

        self.load_ratings()

    def generate(self, topic):
        # Даём модели тему как начало шутки
        prompt = f"<s>{topic} —"
        joke   = generate_joke(self.gpt, self.tokenizer, prompt)
        return joke if joke else "Не смог придумать шутку, попробуй другую тему."

    def learn_from_rating(self, joke, rating):
        words = split_words(joke)
        if rating >= GOOD_RATING:
            change = 0.15
        elif rating <= BAD_RATING:
            change = -0.15
        else:
            change = 0.02
        for word in words:
            self.word_weights[word] += change

    def save_rating(self, topic, joke, rating):
        file_exists = os.path.exists(RATINGS_FILE)
        with open(RATINGS_FILE, "a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["topic", "joke", "rating"])
            writer.writerow([topic, joke, rating])

    def load_ratings(self):
        if not os.path.exists(RATINGS_FILE):
            return
        with open(RATINGS_FILE, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.learn_from_rating(row["joke"], int(row["rating"]))

# ─── Интерфейс ────────────────────────────────────────────────────────────────

def ask_rating():
    while True:
        text = input("Оцени шутку от 1 до 5: ").strip()
        if text in ["выход", "exit", "quit"]:
            return None
        if text in ["1", "2", "3", "4", "5"]:
            return int(text)
        print("Нужно ввести число от 1 до 5.")


def main():
    jokes = load_jokes()
    ai    = JokeAI(jokes)
    print("\nИИ готов. Пиши тему шутки или 'выход'.")

    while True:
        topic = input("\nТема шутки: ").strip().lower()
        if topic in ["выход", "exit", "quit"]:
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")
            break

        joke = ai.generate(topic)
        print("\n" + joke)

        rating = ask_rating()
        if rating is None:
            print("Пока! Оценки сохранены, ИИ стал чуть умнее.")
            break

        ai.learn_from_rating(joke, rating)
        ai.save_rating(topic, joke, rating)
        print("Оценка сохранена. Следующая шутка будет учитывать твой вкус.")


if __name__ == "__main__":
    main()