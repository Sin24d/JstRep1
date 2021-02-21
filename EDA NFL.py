import numpy as np 
import pandas as pd

import seaborn as sns


import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.dpi'] = 150

import cv2
import imageio
from IPython.display import Video, display

import warnings
warnings.filterwarnings('ignore')

train_tracking = pd.read_csv('../input/nfl-impact-detection/train_player_tracking.csv')
test_tracking = pd.read_csv('../input/nfl-impact-detection/test_player_tracking.csv')


train_labels = pd.read_csv('../input/nfl-impact-detection/train_labels.csv')
image_labels = pd.read_csv('../input/nfl-impact-detection/image_labels.csv')
video_labels = pd.read_csv('/kaggle/input/nfl-impact-detection/train_labels.csv')

sub_sample = pd.read_csv('../input/nfl-impact-detection/sample_submission.csv')

train_labels.nunique().to_frame().rename(columns={0:"Count"})

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

sns.distplot(train_labels["gameKey"].value_counts(), ax=ax[0, 0], rug=True, color="red")
ax[0, 0].set_title("Game Counts")

sns.distplot(train_labels["playID"].value_counts(), ax=ax[0, 1], rug=True, color="blue")
ax[0, 1].set_title("Play Counts")

sns.distplot(train_labels["label"].value_counts(), ax=ax[1, 0], rug=True, color="green")
ax[1, 0].set_title("Labels Counts")

sns.distplot(train_labels["video"].value_counts(), ax=ax[1, 1], rug=True, color="yellow")
ax[1, 1].set_title("Videos Counts")

fig.show()

train_labels['video'].nunique()

play_frame_count = train_labels[['gameKey','playID','frame']].drop_duplicates()[['gameKey','playID']].value_counts()

fig, ax = plt.subplots(figsize=(10, 8))
sns.distplot(play_frame_count, bins=15)
ax.set_title('Distribution of frames per video file')
plt.show()
train_labels['area'] = train_labels['width'] * train_labels['height']
fig, ax = plt.subplots(figsize=(10, 5))

sns.distplot(train_labels['area'].value_counts(),
             bins=10)
ax.set_title('Distribution bounding box sizes')
plt.show()

train_labels['impactType'].value_counts().plot(kind='bar',title='Impact Type Count',figsize=(12, 4))

plt.show()

train_labels['impactType'].value_counts()

sns.catplot(x="view", hue="impactType", col="confidence",
                data=train_labels, kind="count")

sns.catplot(x="view", hue="impactType", col="visibility",
                data=train_labels, kind="count")

impact_occ = train_labels[['video','impact']].fillna(0)['impact'].mean() * 100
print(f'Of all bounding boxes, {impact_occ:0.4f}% of them involve an impact event')

train_labels['confidence'].dropna().astype('int').value_counts().plot(kind='bar',
          title='Confidence Type Label Count',
          figsize=(12, 4))
plt.show()

train_labels['confidence'].value_counts()

sns.catplot(x="impactType", hue="confidence", col="view",
                data=train_labels, kind="count")

train_labels['visibility'].dropna() \
    .astype('int').value_counts() \
    .plot(kind='bar',
          title='Visibility Label Count',
          figsize=(12, 4))
plt.show()

train_labels['visibility'].value_counts()

train_labels['visibility'].dropna() \
    .astype('int').value_counts() \
    .plot(kind='bar',
          title='Visibility Label Count',
          figsize=(12, 4))
plt.show()

train_labels['visibility'].value_counts()

