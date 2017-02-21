import csv
import os

import numpy as np
import pandas as pd
from scipy.special import logit
import sklearn
from sklearn.linear_model import (MultiTaskLassoCV, LogisticRegressionCV,
                                  LinearRegression, LogisticRegression)
pd.options.mode.chained_assignment = None

DIR = os.path.dirname(os.path.realpath(__file__))
DATA = os.path.join(DIR, 'RegularSeasonDetailedResults.csv')
TEAM_DATA = os.path.join(DIR, 'Teams.csv')
CLEAN_DATA_DIR = os.path.join(DIR, 'clean_data')

LSUFFIX = '_first'
RSUFFIX = '_second'

if not os.path.isdir(CLEAN_DATA_DIR):
    os.mkdir(CLEAN_DATA_DIR)


def get_filename(year):
    if year is not None:
        return os.path.join(CLEAN_DATA_DIR, '{}.csv'.format(year))
    else:
        return os.path.join(CLEAN_DATA_DIR, 'all.csv')


class TeamCache(object):
    def __init__(self):
        self._id_to_team = {}
        self._team_to_id = {}
        self._loaded = False

    def _load(self):
        if not self._loaded:
            with open(TEAM_DATA, 'r') as buff:
                for row in csv.DictReader(buff):
                    self._id_to_team[int(row['Team_Id'])] = row['Team_Name']
                    self._team_to_id[row['Team_Name']] = int(row['Team_Id'])
        self._loaded = True

    def id_to_team(self, id_):
        self._load()
        return self._id_to_team.get(int(id_))

    def team_to_id(self, team):
        self._load()
        return self._team_to_id.get(team)

    def find_team(self, team):
        self._load()
        if team in self._team_to_id:
            return team
        matches = [t for t in self._team_to_id if team.lower() in t.lower()]
        if matches:
            return ', '.join(matches)
        return 'No matches found'

    def check_teams(self, *teams):
        self._load()
        for team in teams:
            if team not in self._team_to_id:
                raise LookupError(self.find_team(team))


TEAM_CACHE = TeamCache()


def data_gen(year):
    if year is None:
        def row_filter(row):
            return True
    else:
        year = str(year)

        def row_filter(row):
            return row['Season'] == year

    with open(DATA, 'r') as buff:
        for idx, row in enumerate(csv.DictReader(buff)):
            if row_filter(row):
                for letter in ('W', 'L'):
                    data = {
                        'game_id': idx,
                        'won': letter == 'W',
                        'day_num': int(row['Daynum']),
                        'season': int(row['Season']),
                        'num_ot': int(row['Numot']),
                        'home_game': row['Wloc'] == {'W': 'H', 'L': 'A'}[letter],
                        'team_name': TEAM_CACHE.id_to_team(row[letter + 'team']),
                    }
                    for key, value in row.items():
                        if key.startswith(letter) and key != 'Wloc':
                            data[key[1:]] = int(value)
                    yield data


def rolling_avg(group, col, min_periods=5):
    return group[col].shift(1).expanding(min_periods=min_periods).mean()


def rolling_sum(group, col, min_periods=5):
    return group[col].shift(1).expanding(min_periods=min_periods).sum()


def get_df(year):
    return pd.DataFrame(list(data_gen(year)))


def gen_features(df, min_periods=5):
    avg_features = [
        'ast',
        'blk',
        'dr',
        'or',
        'pf',
        'score',
        'stl',
        'to',
        'won']

    sum_features = [
        'fga',
        'fga3',
        'fgm',
        'fgm3',
        'fta',
        'ftm',
        ]

    def transformer(group):
        transformed = {'avg_{}'.format(x): rolling_avg(group, x, min_periods) for x in avg_features}
        transformed.update(
            {'tot_{}'.format(x): rolling_sum(group, x, min_periods) for x in sum_features}
        )
        transformed.update(group.to_dict())
        return pd.DataFrame(transformed)
    features = df.groupby(['season', 'team']).apply(transformer).dropna()
    features['fg_pct'] = features.tot_fgm / np.maximum(features.tot_fga, 1)
    features['fg3_pct'] = features.tot_fgm3 / np.maximum(features.tot_fga3, 1)
    features['ft_pct'] = features.tot_ftm / np.maximum(features.tot_fta, 1)
    return features.reset_index(drop=True)


def get_training_data(features):
    win_first = features[features.won].join(features[~features.won].set_index('game_id'),
                                            on='game_id', how='inner',
                                            lsuffix=LSUFFIX, rsuffix=RSUFFIX)
    lose_first = features[~features.won].join(features[features.won].set_index('game_id'),
                                              on='game_id', how='inner',
                                              lsuffix=LSUFFIX, rsuffix=RSUFFIX)
    return sklearn.utils.shuffle(pd.concat([win_first, lose_first])).reset_index(drop=True)


def get_predict_data(df):
    df.loc[:, 'game_id'] = 0
    df = pd.DataFrame(df.iloc[[0]]).join(pd.DataFrame(df.iloc[[1]]).set_index('game_id'),
                                         on='game_id', how='inner',
                                         lsuffix=LSUFFIX, rsuffix=RSUFFIX).reset_index(drop=True)
    df.loc[0, 'home_game' + LSUFFIX] = True
    df.loc[0, 'home_game' + RSUFFIX] = False
    return df


def get_clean_data(year=None):
    filename = get_filename(year)
    if not os.path.exists(filename):
        gen_features(get_df(year)).to_csv(filename, index=False)
    return pd.read_csv(filename)


def regression_target(df):
    return df[['score' + LSUFFIX, 'score' + RSUFFIX]]


def classification_target(df):
    return df['score' + LSUFFIX] > df['score' + RSUFFIX]


def get_feature_names():
    base_features = ['avg_score', 'fg_pct', 'fg3_pct', 'ft_pct', 'avg_or', 'avg_dr',
                     'avg_to', 'avg_stl', 'avg_won', 'home_game']
    features = []
    for feature in base_features:
        if feature.endswith('_pct') or feature == 'avg_won':
            feature = 'logit_' + feature
        for suffix in (LSUFFIX, RSUFFIX):
            features.append(feature + suffix)
    return features


def get_features(df):
    features = df[[f for f in get_feature_names() if not f.startswith('logit_')]]
    epsilon = 1e-6
    for feature in [f for f in get_feature_names() if f.startswith('logit_')]:
        features[feature] = logit(df[feature[6:]].clip(epsilon, 1-epsilon))
    return features


def get_regression_model_class(cv):
    if cv:
        return MultiTaskLassoCV(fit_intercept=False)
    return LinearRegression(fit_intercept=False)


def get_classification_model_class(cv):
    if cv:
        return LogisticRegressionCV(fit_intercept=False)
    else:
        return LogisticRegression(fit_intercept=False)


def fit_regression_model(df, cv=False):
    return get_regression_model_class(cv).fit(get_features(df), regression_target(df))


def fit_classification_model(df, cv=False):
    return get_classification_model_class(cv).fit(get_features(df), classification_target(df))


def get_team_predict_data(df, team_one, team_two):
    teams = (team_one, team_two)
    TEAM_CACHE.check_teams(*teams)
    last_games = df.iloc[[df.day_num[df.team_name == team].idxmax() for team in teams]]
    return get_features(get_predict_data(last_games))


def get_historical_data():
    df = get_clean_data()
    df = df[df.season != 2016]
    return get_training_data(df)

# PUBLIC API
LATEST_DATA = get_clean_data(2016)


def get_models(cv=False):
    df = get_historical_data()
    return fit_regression_model(df, cv=cv), fit_classification_model(df, cv=cv)


def predict_scores(reg, team_one, team_two):
    teams = (team_one, team_two)
    data = get_team_predict_data(LATEST_DATA, *teams)
    scores = reg.predict(data).round().astype(int)
    msg = []
    for team, score in zip(teams, scores[0]):
        msg.append('{} {}'.format(team, score))
    return ' '.join(msg)


def predict_winner(clf, team_one, team_two):
    teams = (team_one, team_two)
    data = get_team_predict_data(LATEST_DATA, *teams)
    prob = int(round(clf.predict_proba(data)[0][1] * 100))
    return '{} has a {:d}% chance of beating {}'.format(team_one, prob, team_two)


def predict(reg, clf, team_one, team_two):
    winner_msg = predict_winner(clf, team_one, team_two)
    score_msg = predict_scores(reg, team_one, team_two)
    print("{}\nPredicted Score:\n\t{}".format(winner_msg, score_msg))


def explain_model(reg):
    msg = ['predicted_score_first = ']
    for coef, feature in zip(reg.coef_[0], get_feature_names()):
        msg.append('\t{:+.2f} * {}'.format(coef, feature))
    print('\n'.join(msg))
