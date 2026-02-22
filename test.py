import json
from urllib.request import urlopen, Request
url = 'https://gamma-api.polymarket.com/events?active=true&closed=false&tag_id=102892&ascending=true&limit=20'
req = Request(url, headers={'User-Agent': 'test/1.0'})
with urlopen(req, timeout=10) as r:
    data = json.loads(r.read())
print('Events found:', len(data))
for e in data:
    markets = e.get('markets') or []
    for m in markets:
        print(m.get('question','')[:60], '|', m.get('endDate',''), '| accepting:', m.get('acceptingOrders'))
