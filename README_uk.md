# 🍳 Price Picker 🔻

Інструмент для пошуку символів на Binance Futures

[👉 Англійська версія](README.md)

## Приклади використання

### Загальний пошук

🎩 Приклад. Знайти символи з волатильністю щонайменше 2% на 4-годинному таймфреймі

```bash
  python picker.py --interval=15m --range=4h --threshold=2%
```

<img width="1002" alt="image" src="https://github.com/user-attachments/assets/20000cd0-c663-4ab1-b70e-3e817c1e5fab">

### Пошук для коротких позицій

Приклад. Знайти символи для коротких позицій (ближче до верхнього краю) з волатильністю щонайменше 3% на 2-годинному таймфреймі.

```bash
 python picker.py --interval=5m --range=2h --threshold=3% --short 
```

<img width="1026" alt="image" src="https://github.com/user-attachments/assets/5c6951af-d4ba-40e5-8ea1-1511aa6ff17d">

### Пошук для довгих позицій

Приклад. Знайти символи для довгих позицій (ближче до нижнього краю) з волатильністю щонайменше 5% на 4-годинному таймфреймі.

```bash
 python picker.py --interval=5m --range=4h --threshold=5% --long 
```

<img width="1120" alt="image" src="https://github.com/user-attachments/assets/010730b1-1e00-4453-bd3c-7a06659e680a">

## Встановлення

1. Переконайтеся, що Python встановлено на вашому пристрої

Приклад встановлення на Ubuntu Linux:
```bash
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python-is-python3
```

Приклад встановлення на Android (Termux):
```bash
pkg update && pkg upgrade -y && pkg install -y python
```

2. Завантажте скрипт на свій пристрій<br>

```bash
# Завантажте скрипт з репозиторію
curl -O https://raw.githubusercontent.com/asidko/binance-picker/main/picker.py
# ☝️ Повторіть цю команду пізніше, якщо бажаєте оновити скрипт до новішої версії
```

3. Встановіть необхідні пакети Python

```bash
pip install aiohttp rich
```

4. Запустіть скрипт (див. приклади використання вище)

```bash
 python picker.py --interval=5m --range=2h --threshold=2%
```

## Спеціальні параметри

### --help

Приклад: `python picker.py --help`

Перегляд всіх доступних опцій

### --watch

Приклад: `python picker.py --interval=5m --range=2h --threshold=4% --watch`

Автоматично отримує нові дані кожні 30 секунд і відображає їх

Ви можете змінити інтервал, передавши `--wait=300` (у секундах) для запиту даних кожні 5 хвилин

## Ліцензія

Цей проєкт ліцензовано за умовами ліцензії MIT - дивіться файл [LICENSE](LICENSE) для деталей