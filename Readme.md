[Khan Academy Calculus](https://www.khanacademy.org/math/calculus-1?utm_source=chatgpt.com)

# 🤖 Joke AI (from scratch)

Проект: создание собственной нейросети для генерации шуток  
Без использования готовых LLM — только собственная модель, токенизация и обучение.

---

# 🎯 Цель

Создать систему, которая:

- обучается на шутках и анекдотах
- понимает структуру юмора
- генерирует новые шутки
- работает полностью локально

---

# 🧠 Общий план разработки

---

## 1. Основы нейросетей

Что нужно понять:

- нейрон
- функции активации
- градиентный спуск
- backpropagation

### 📚 Материалы:
- https://www.3blue1brown.com/lessons/neural-networks  
- https://cs231n.github.io/optimization-2/  
- http://neuralnetworksanddeeplearning.com/  

---

## 2. Реализация нейросети на NumPy

Цель — написать всё вручную:

- Dense layer
- ReLU / Softmax
- Loss function
- Backpropagation
- SGD optimizer

### 📚 Материалы:
- https://numpy.org/doc/stable/  
- https://cs231n.github.io/neural-networks-case-study/  

---

## 3. Токенизация текста

Нужно создать свой токенизатор:

Функции:
- split text → tokens
- build vocabulary
- encode / decode

Пример:

"кот пошёл домой"
→ [12, 98, 451]


### 📚 Материалы:
- https://huggingface.co/docs/tokenizers  
- https://leimao.github.io/blog/BPE-Algorithm/  

---

## 4. Датасет шуток

Формат:

```json
{
  "text": "Программист купил 0.5 молока, потому что больше не требовалось."
}

Источники:

анекдоты
мемы
стендап
короткие истории

Важно:

избегать дублей
короткие тексты работают лучше
разнообразие тем
5. Простая языковая модель (LSTM)

Архитектура:

Embedding → LSTM → Linear → Softmax

Файлы:

model/
  embedding.py
  lstm.py
  model.py

training/
  dataset.py
  trainer.py
📚 Материалы:
https://pytorch.org/tutorials/
https://colah.github.io/posts/2015-08-Understanding-LSTMs/
6. Обучение модели

Задача:

predict next token

Пример:

"кот зашёл в" → "дом"

Методы:

cross entropy loss
Adam optimizer
batching
shuffling
7. Генерация текста

Алгоритм:

seed → predict next token → append → repeat

Дополнительно:

temperature sampling
top-k sampling
8. Логика юмора (надстройка)

Идеи:

неожиданное продолжение
нарушение ожиданий
абсурд
игра слов

Можно добавить:

HumorScore(text)
9. Переход к Transformer (опционально)

Архитектура:

Embedding → Self-Attention → Feed Forward → Output
📚 Материалы:
https://arxiv.org/abs/1706.03762
https://jalammar.github.io/illustrated-transformer/
10. Структура проекта
joke-ai/

data/
  jokes.json

model/
  tokenizer.py
  lstm.py
  embedding.py

training/
  dataset.py
  trainer.py

inference/
  generate.py

utils/
  text_utils.py

main.py
🚀 Итог

После завершения ты получишь:

свою языковую модель
свой токенизатор
свой датасет
генератор шуток
понимание как работают LLM изнутри
⚡ Рекомендация

Начни с:

NumPy → LSTM → генерация текста

И только потом переходи к Transformer.


---