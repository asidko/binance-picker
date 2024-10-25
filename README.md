# 🍳 Price Picker 🔻

Binance Futures symbol lookup tool

[👉 Ukrainian version](README_uk.md)

## Usage examples

### General lookup

🎩Example. Find symbols with at least 2% volatility on the 4 hours timeframe

```bash
  python picker.py --interval=15m --range=4h --threshold=2%
```

<img width="1002" alt="image" src="https://github.com/user-attachments/assets/20000cd0-c663-4ab1-b70e-3e817c1e5fab">

### Short positions lookup

Example. Find symbols for short positions (closer to top edge) with at least 3% volatility on the 2 hours timeframe.

```bash
 python picker.py --interval=5m --range=2h --threshold=3% --short 
```

<img width="1026" alt="image" src="https://github.com/user-attachments/assets/5c6951af-d4ba-40e5-8ea1-1511aa6ff17d">

### Long positions lookup

Example. Find symbols for long positions (closer to bottom edge) with at least 5% volatility on the 4 hours timeframe.

```bash
 python picker.py --interval=5m --range=4h --threshold=5% --long 
```

<img width="1120" alt="image" src="https://github.com/user-attachments/assets/010730b1-1e00-4453-bd3c-7a06659e680a">


## Installation

1. Make sure python is installed on your machine

Example of installation on Ubuntu Linux:
```bash
sudo apt-get update -y && sudo apt-get install -y python3 python3-pip python-is-python3
```

Example of installation on Android (Termux):
```bash
pkg update && pkg upgrade -y && pkg install -y python
```

2. Download the script to your machine<br>

```bash
# Download the script form the repository
curl -O https://raw.githubusercontent.com/asidko/binance-picker/main/picker.py
# ☝️ Repeat this command later if you want to update the script to a newer version
```

3. Install required python packages

```bash
pip install aiohttp rich
```

4. Run the script (check the usage examples above)

```bash
 python picker.py --interval=5m --range=2h --threshold=2%
```

## Special params

### --help

Example: `python picker.py --help`

See all available options

### --watch

Example: `python picker.py --interval=5m --range=2h --threshold=4% --watch`

Automatically request new data every 30 seconds and show it

You can change the interval by passing `--wait=300` (in seconds) to request data every 5 minutes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details