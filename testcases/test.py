import time

def get_event_slug(coin="btc"):
    ts = int(time.time() // 300) * 300   # 5-minute rounding
    return f"{coin}-updown-5m-{ts}"

for coin in ["btc", "eth", "sol", "xrp"]:
    slug = get_event_slug(coin)
    url = f"https://polymarket.com/event/{slug}"
    print(url)