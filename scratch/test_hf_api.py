import requests
url = "https://datasets-server.huggingface.co/filter?dataset=phreshphish/phreshphish&config=default&split=train&where=label='phishing'&offset=0&length=10"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    rows = data.get('rows', [])
    print(f"Got {len(rows)} filtered rows.")
    for r in rows[:2]:
        print("Label:", r['row']['label'])
else:
    print("Error:", response.status_code, response.text)
