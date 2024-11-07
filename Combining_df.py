import pandas as pd


season_21 = pd.read_csv('data/pitches_21.csv')
season_22 = pd.read_csv('data/pitches_22.csv')
season_23 = pd.read_csv('data/pitches_23.csv')

combined_df = pd.concat([season_21, season_22, season_23])

# Drop rows with any NAs
cleaned_df = combined_df.dropna()

# Converting GameDate to pandas Time Objects
cleaned_df['GameDate'] = pd.to_datetime(cleaned_df['GameDate'])
cleaned_df['Date'] = cleaned_df['GameDate'].dt.date
cleaned_df['Time'] = cleaned_df['GameDate'].dt.time

cols = list(cleaned_df.columns)
# find index of GameDate
idx = cols.index('GameDate')
# Basically put Date and Time right next to GameDate
new_cols = cols[:idx + 1] + ['Date', 'Time'] + cols[idx + 1:-2]

cleaned_df = cleaned_df[new_cols].copy()


cleaned_df  = cleaned_df.sort_values(by = ['Date', 'Home', 'ab'])
cleaned_df = cleaned_df.drop(columns = ['Unnamed: 0'])


cleaned_df = cleaned_df.sort_values(by=['gameid', 'ab', 'pitchnum'])

print(cleaned_df.head(30))

cleaned_df.to_pickle('data/aggregated_data.pkl')


