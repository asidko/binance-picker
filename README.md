# 🍳 Price Picker 🔻

Binance Futures symbol lookup tool

[👉 Ukrainian version](README_uk.md)

## Usage examples

🎩Example. Find symbols with 2% volatility on the 2 hours timeframe and show how close is the current price to the nearest level

```bash
  python picker.py --interval=5m --range=2h --threshold=2%
```

![image](https://github.com/user-attachments/assets/a0903a3a-8e6c-4006-9efa-ae08c912c3e1)

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