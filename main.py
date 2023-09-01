import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# nltk.download()
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')
# need to be installed pip install ipywidgets
plt.style.use('ggplot')


# Reading data
df1 = pd.read_csv('datasets/reviews.csv')
df2 = pd.read_csv('datasets/labels.csv')

# print(df1.head())
# print(df2.head())
#

merged_df = pd.merge(df1, df2, on='id')
merged_df.to_csv('merged.csv', index=False)

# print(merged_df.head())

ax = merged_df['sentiment'].value_counts().sort_index().plot(kind='bar', title='Amount of Positive and Negative reviews', figsize=(10, 8.5))
ax.set_xlabel('Sentiments')
# plt.show()

example = merged_df['text'][10]
# print(example)
# tokens = nltk.word_tokenize(example)
# print(tokens[:10])
#
# tagged =nltk.pos_tag(tokens)
# print(tagged[:10])
#
# entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()

sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores('I am so happy!'))
print(sia.polarity_scores('This is the worst thing ever. I hate it'))

print(sia.polarity_scores(example))

# Run the polarity score

res ={}
for i, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    text=row['text']
    myid=row['id']
    res[myid]=sia.polarity_scores(text)

# print(res)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'id'})
vaders=vaders.merge(merged_df, how='left')
print(vaders.head())

# ax1= sns.barplot(data=vaders, x= 'sentiment', y='compound')
# ax1.set_title('Compaund Sentiment')
# plt.show()
ax1= sns.barplot(data=vaders, x= 'sentiment', y='neg')

plt.show()