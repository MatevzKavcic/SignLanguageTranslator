import numpy as np

y = np.load("../TrySeven/embeddingsNpy/embeddings.npy", allow_pickle=True)

print(y[:20])
print(np.unique(y)[:20])
print("max =", np.max(y))
print("min =", np.min(y))
