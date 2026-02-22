import sys

import pandas as pd
import numpy as np
from holoviews.annotators import preprocess
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Just to identify different steps
def step(msg=''):
    dash = 40
    print(f'[{msg}]', '-' * dash, sep='')
    print('-' * (dash + len(msg) + 2))


# load data
def load_data():
    try:
        return pd.read_excel('dataset_pretraitement_fraude.xlsx')
    except Exception as e:
        print(f'Erreur de chargement {e}')
        sys.exit(0)


def remove_columns(df):
    # On enleve TransactionID, ClientID, Commentaire
    # print('revalidating df')
    columns = ['TransactionID', 'ClientID', 'Commentaire']
    # print('Dropping', columns)
    return remove_TODOs_columns(df.drop(columns=columns))


def remove_TODOs_columns(df):
    return df.loc[:, ~df.columns.str.startswith('TODO_')]


def build_X_y(df):
    X = df.drop(columns=['Fraude'])
    y = df['Fraude']

    return X, y


def show(X, row):
    step(f"Showing '{row}' first line of X variables")
    print(X.head(row).to_string())


# Preprocessing...

def fix_dates(X):
    col = X.columns[0]

    raw = (
        X[col]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace("INCONNU", pd.NA)
    )

    dates = pd.to_datetime(raw, format='mixed', errors='coerce')

    df = pd.DataFrame({
        "jour_semaine": dates.dt.weekday,
        "mois": dates.dt.month,
        "date_invalide": dates.isna().astype(int)
    })

    return df


def fix_montant(X):
    col = X.columns[0]
    montant = X[col].copy()

    # transform into upper string
    montant = montant.astype('string').str.strip().str.upper()

    # identifying conversions needed
    is_k = montant.str.contains('K', na=False)
    is_usd = montant.str.contains('USD', na=False)

    # removing disturbing characters
    montant = montant.str.replace(r'[A-Z,\s]', '', regex=True)

    # convert into numeric
    montant = pd.to_numeric(montant, errors='coerce')

    # apply the conversions
    montant = montant.where(~is_usd, montant * 130)
    montant = montant.where(~is_k, montant * 1000)

    return pd.DataFrame(montant)

def normalize_devise(X):
    col = X.columns[0]
    devise = (
        X[col]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace('USD', 'HTG', regex=False)
    )

    return devise.to_frame()


if __name__ == '__main__':
    import os

    print(os.listdir(os.getcwd()))  # listing files
    step('Loadimg data')
    df = load_data()
    print('data loaded successfully')
    # shape (618, 27)
    df = remove_columns(df)
    df.drop_duplicates(inplace=True) #removing duplicated line, shape (600, 16)

    sys.exit(0)

    X, y = build_X_y(df)

    preprocessor = ColumnTransformer(
        transformers=[
            ('date', Pipeline([
                ('extract_date_features', FunctionTransformer(fix_dates)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), ['DateTransaction_raw']),
            ('montant', Pipeline([
                ('convert', FunctionTransformer(fix_montant)),
                ('impute', SimpleImputer(strategy='median')),
                ('scale', StandardScaler())
            ]), ['Montant_raw']),
            ('devise', Pipeline([
                ('normalize', FunctionTransformer(normalize_devise)),
                ('encoder', OneHotEncoder())
            ]), ['Devise_indiquee'])
        ], verbose=True
    )

    full_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=100))
    ])

    step('printing full model')
    print(full_model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
        # shuffle=False
    )

    skf = StratifiedKFold(n_splits=5 ,shuffle=True, random_state=42)
    cv_scores = cross_val_score(full_model, X_train, y_train, cv=skf, scoring='accuracy')

    # fix_montant(pd.DataFrame(X['Montant_raw']))