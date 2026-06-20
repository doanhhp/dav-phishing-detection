from datasets import load_dataset

print("\nLoading ealvaradob...")
try:
    ds2 = load_dataset("ealvaradob/phishing-dataset", "webs", split="train", streaming=True, trust_remote_code=True)
    sample2 = next(iter(ds2))
    print("ealvaradob columns:", list(sample2.keys()))
    print("ealvaradob sample text prefix:", str(sample2.get("text", ""))[:200])
except Exception as e:
    print("Error loading ealvaradob:", e)
