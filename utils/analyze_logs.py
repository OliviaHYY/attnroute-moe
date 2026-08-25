import json, sys, numpy as np
with open(sys.argv[1]) as f:
    h = json.load(f)['history']
print(f"Epochs: {len(h)}")
print(f"CV@ep1:  {h[0]['cv']:.4f}")
print(f"CV@ep5:  {h[4]['cv']:.4f}")
print(f"CV@final:{h[-1]['cv']:.4f}")
print(f"Best acc:{max(e['val_acc'] for e in h)*100:.2f}%")
