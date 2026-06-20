from datasets import load_dataset
import traceback

print("Loading PhreshPhish...")
try:
    ds1 = load_dataset("phreshphish/phreshphish", split="train", streaming=True)
    sample1 = next(iter(ds1))
    print("Success:", list(sample1.keys()))
except Exception as e:
    print("Error loading PhreshPhish:")
    traceback.print_exc()
