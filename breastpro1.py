#0 = malignant
#1 = benign

import sklearn.datasets
breast_data = sklearn.datasets.load_breast_cancer()

import pandas as pd
df = pd.DataFrame(data=breast_data.data, columns=(breast_data.feature_names))

df['Class'] = breast_data.target
#Target = Class

import matplotlib.pyplot as plt
plt.hist(df['Class'])
plt.xlabel('Class')
plt.ylabel('Instances')
print(plt.show())


from matplotlib import pyplot as plt
fig2, axes = plt.subplots(nrows=3, ncols=4, figsize=(15,15))
for i, ax in enumerate(axes.flatten()):
    if i > 9:
        break
    ax.set_ylabel("Class")
    ax.set_xlabel(df.columns[i])
    ax.plot(df[df.columns[i]], breast_data.target, 'o')
plt.show()
