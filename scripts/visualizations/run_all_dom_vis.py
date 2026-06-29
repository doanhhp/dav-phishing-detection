import os
import subprocess
import sys

def main():
    datasets = ['Main', 'OOD', 'PhreshPhish']
    configs = [
        {'depth': 5, 'min_freq': 0.02},
        {'depth': 10, 'min_freq': 0.01},
        {'depth': 20, 'min_freq': 0.005}
    ]
    samples = 1000

    print("Starting batch generation of DOM tree visualizations...")
    for d in datasets:
        for c in configs:
            cmd = [
                ".venv\\Scripts\\python.exe", 
                "scripts/visualizations/visualize_dom_tree.py",
                "--dataset", d,
                "--depth", str(c['depth']),
                "--min-freq", str(c['min_freq']),
                "--samples", str(samples)
            ]
            print(f"\n--- Running: {' '.join(cmd)} ---")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command for {d} at depth {c['depth']}: {e}")
                sys.exit(1)
                
    print("\nAll visualizations completed successfully!")

if __name__ == "__main__":
    main()
