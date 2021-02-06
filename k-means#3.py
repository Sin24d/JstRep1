import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data = pd.read_csv('Desktop/coledge.csv',index_col=0)
data.head()

data.info()

data.describe()

sns.set_style('whitegrid')
sns.lmplot('Grad.Rate','Room.Board',data,hue='Private',size=8,fit_reg=False,aspect=1,palette='coolwarm')
sns.lmplot('Outstate','F.Undergrad',data,hue='Private',size=8,fit_reg=False,aspect=1,palette='coolwarm')
g = sns.FacetGrid(data,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=1)
g = sns.FacetGrid(data,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=1)
data[data['Grad.Rate']>100]

sns.set_style('darkgrid')
g = sns.FacetGrid(data,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=1)
##############
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2)
kmeans.fit(data.drop('Private',axis=1))
kmeans.cluster_centers_

def convert(label):
    if label == 'Yes':
        return 1
    else:
        return 0
data['Cluster'] = data['Private'].apply(convert)
data.head()
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(data['Cluster'],kmeans.labels_))
print(classification_report(data['Cluster'],kmeans.labels_))
##jupyter

