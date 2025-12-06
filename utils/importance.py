import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_importance(x_train, y_train, shuffle=False, seed=42):
    if shuffle:
        y_train = y_train.sample(frac=1.0, random_state=np.random.randint(0, 999))

    model = lgb.train(
        params={
            'objective': 'binary',
            'boosting_type': 'rf',
            'subsample': 0.623,
            'colsample_bytree': 0.7,
            'num_leaves': 127,
            'max_depth': 8,
            'seed': seed,
            'baggingfeature_namereq': 1,
            'n_jobs': 4,
            "verbosity": -1,
        },
        train_set=lgb.Dataset(x_train, y_train, free_raw_data=False),
        num_boost_round=200
    )

    return pd.DataFrame({
        "feature": x_train.columns,
        "importance_gain": model.feature_importance(importance_type='gain'),
        "importance_split": model.feature_importance(importance_type='split'),
    })


def get_null_importance(x_train, y_train, runs=80):
    null_importance_df = pd.DataFrame()

    for i in tqdm(range(runs)):
        importance_df = get_importance(x_train, y_train, shuffle=True)
        importance_df['run'] = i + 1
        null_importance_df = pd.concat([null_importance_df, importance_df], axis=0)

    return null_importance_df


def display_distributions(importance_df, null_importance_df, feature_name):
    plt.figure(figsize=(13, 6))

    # Plot Split Importance
    ax = plt.subplot(1, 2, 1)
    a = ax.hist(
        null_importance_df.loc[null_importance_df['feature'] == feature_name, 'importance_split'].values,
        label='Null Importance',
    )
    ax.vlines(
        x=importance_df.loc[importance_df['feature'] == feature_name, 'importance_split'].mean(),
        ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target'
    )
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_name.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_name.upper())

    # Plot Gain importance
    ax = plt.subplot(1, 2, 2)
    a = ax.hist(
        null_importance_df.loc[null_importance_df['feature'] == feature_name, 'importance_gain'].values,
        label='Null Importance'
    )
    ax.vlines(
        x=importance_df.loc[importance_df['feature'] == feature_name, 'importance_gain'].mean(),
        ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target'
    )
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_name.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_name.upper())


def get_importance_score(x_train, y_train, seed=42):
    np.random.seed(seed)
    x_new = x_train.copy()
    columns_map = {v: f"v{k}" for k, v in enumerate(x_new.columns)}
    columns_map_reverse = {v: k for k, v in columns_map.items()}
    x_new = x_new.rename(columns=columns_map)

    print("计算Importance")
    importance_df = get_importance(x_new, y_train, seed=seed)
    importance_df["feature"] = importance_df["feature"].apply(lambda x: columns_map_reverse[x])

    print("计算Null Importance")
    null_importance_df = get_null_importance(x_new, y_train)
    null_importance_df["feature"] = null_importance_df["feature"].apply(lambda x: columns_map_reverse[x])

    feature_scores = []
    for feature_name in importance_df['feature'].tolist():
        importance_mask = importance_df['feature'] == feature_name
        null_importance_mask = null_importance_df['feature'] == feature_name

        null_gain = null_importance_df.loc[null_importance_mask, 'importance_gain'].values
        gain = importance_df.loc[importance_mask, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + gain / (1 + np.percentile(null_gain, 75)))

        null_split = null_importance_df.loc[null_importance_mask, 'importance_split'].values
        split = importance_df.loc[importance_mask, 'importance_split'].mean()
        split_score = np.log(1e-10 + split / (1 + np.percentile(null_split, 75)))

        feature_scores.append((feature_name, split_score, gain_score))

    score_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    score_df["split_rank"] = score_df["split_score"].rank()
    score_df["gain_rank"] = score_df["gain_score"].rank()
    score_df["rank"] = (score_df["split_rank"] + score_df["gain_rank"]) / 2
    score_df = score_df.rename(columns=columns_map_reverse)
    score_df = score_df.sort_values('rank', ascending=False)

    return score_df, importance_df, null_importance_df
