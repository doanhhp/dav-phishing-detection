import requests
url = "https://datasets-server.huggingface.co/info?dataset=phreshphish/phreshphish"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    splits = data.get('dataset_info', {}).get('default', {}).get('splits', {})
    for split, info in splits.items():
        print(f"Split {split} has {info.get('num_examples')} examples.")
else:
    print("Error", response.status_code)
