import numpy as np
import metricspace as ms
distances = np.random.uniform(1, 12, size=(10, 10))
distances = np.array(np.triu(distances, k=1) + np.triu(distances, k=1).T)  # make it symmetric and zero diagonal
labels = ["A", "A", "A", "A", "B", "B", "C", "C", "C", "C"]
nsam = np.unique(labels, return_counts=True)[1]
print(nsam)
confusion_matrix = ms.distclust(distances, nsam)
row_sums = np.sum(confusion_matrix, axis=1)
print(np.array_equal(row_sums, nsam))

print(confusion_matrix.shape, len(nsam)) # these should all be the same!


