import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def clean(df):
    df['date'] = pd.to_datetime(df['date'], utc = True)
    df = df.reindex(df.date.sort_values().index)
    df = df.set_index(df.date)
    print(f"earliest = {df.iloc[0]}")
    print(f"latest = {df.iloc[-1]}")
    #df["count"] = df["count"] / df["count"].iloc[0]
    df = df.groupby(by=[df.index.week]).sum()
    mean = df["count"].mean()
    print(f"mean = {mean}, max = {df['count'].max()} min = {df['count'].min()}")
    df["count"] = df["count"] / mean
    print(df.iloc[-20:-1])
    return df


def rebase(df):
    return df


def load_and_parse(file_name: str):
    fqn = f"/home/henryp/Documents/CandF/{file_name}"
    print(f"Loading {fqn}")
    df = pd.read_csv(fqn, sep="\t", parse_dates=['date'],index_col=['date'])
    df
    return df


def load(file_name: str):
    fqn = f"/home/henryp/Documents/CandF/{file_name}"
    print(f"Loading {fqn}")
    df = pd.read_csv(fqn, sep="\t")
    return df


if __name__ == '__main__':
    #df = load_and_parse("ecds.tsv")
    ecds = clean(load("ecds.tsv"))
    ecds = ecds.rename(columns={'count': 'all'})
    peri = clean(load("myocarditis_pericarditis_ecds.tsv"))
    peri = peri.rename(columns={'count': 'peri_myo'})
    heart = clean(load("heart_block_failure_and_attach.tsv"))
    heart = heart.rename(columns={'count': 'heart_block_congestion_attack'})
    #print(ecds)
    #print(peri)
    df = pd.merge(ecds, heart, how='inner', left_index=True, right_index=True)
    df = pd.merge(df, peri, how='inner', left_index=True, right_index=True)
    #df = pd.merge(heart, peri, how='inner', left_index=True, right_index=True)

    fig, ax = plt.subplots(figsize=(12,8))
    df.plot(ax=ax,legend=True)
    #df.plot(x='date_x', y=['count_x', 'count_y']).plot(figsize=(12,8))
    plt.show()

    #sns.lineplot(x='date',y='count',hue='year',data=df)
    #sns.lineplot(x='date',y='count',data=df)

