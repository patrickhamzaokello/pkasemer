import json
from urllib.request import urlopen, Request
url = 'https://gamma-api.polymarket.com/events?active=true&closed=false&limit=100&ascending=true'
req = Request(url, headers={'User-Agent': 'fastloop/1.0'})
with urlopen(req, timeout=15) as r:
    data = json.loads(r.read())
count = 0
for event in data:
    markets = event.get('markets') or []
    for m in markets:
        q = (m.get('question') or '').lower()
        slug = m.get('slug','')
        if 'bitcoin up or down' in q:
            count += 1
            end = m.get('endDate','')
            start = m.get('eventStartTime') or m.get('startTime','')
            accepting = m.get('acceptingOrders')
            active = m.get('active')
            vol = m.get('volume','0')
            print(str(count) + '.', m.get('question','')[:55])
            print('   endDate:', end, '| eventStartTime:', start)
            print('   active:', active, '| acceptingOrders:', accepting, '| vol:', vol)
            print()
print('Total:', count)