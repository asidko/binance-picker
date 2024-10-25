# 🍳 Price Picker 🔻

Інструмент для пошуку символів на Binance Futures

[👉 Українська версія](README_uk.md)

## Приклади використання

🎩 Приклад. Знайти символи з волатильністю 2% на 2-годинному таймфреймі та дізнатись, наскільки поточна ціна наближена до найближчого рівня

```bash
  python picker.py --interval=5m --range=2h --threshold=2%
```

![image](https://github.com/user-attachments/assets/a0903a3a-8e6c-4006-9efa-ae08c912c3e1)

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