import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Just to identify different steps
def step(msg=''):
    dash = 40
    print(f'[{msg}]', '-' * dash, sep='')
    print('-' * (dash + len(msg) + 2))


# load data
def load_data():
    print('loading data')
    try:
        return pd.read_excel('dataset_pretraitement_fraude.xlsx')
    except Exception as e:
        print(f'Erreur de chargement {e}')
        sys.exit(0)  # TODO add an interactive mode if I got enough time, hopefully :)


def remove_columns(df):
    # On enleve TransactionID, ClientID, Commentaire
    print('revalidating df')
    columns = ['TransactionID', 'ClientID', 'Commentaire']
    print('Dropping', columns)
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


# Clean up

def fix_dates(X):
    X['DateTransaction_raw'] = X['DateTransaction_raw'].astype(str).str.strip()
    X['DateTransaction_raw'] = pd.to_datetime(X['DateTransaction_raw'], format='mixed', errors='coerce')
    return X


def fix_montant(X):
    montant = X['Montant_raw']
    # transform into upper string
    montant = montant.astype('string').str.strip()
    montant = montant.str.upper()

    # identifying the ones that contains k te multiply them the by 1000
    is_k = montant.str.contains('K', na=False)
    is_usd = montant.str.contains('USD', na=False)

    # removing disturbing characters
    montant = montant.str.replace(r'[A-Z,\s]', '', regex=True)

    # convert into numeric so we can do math operations
    montant = pd.to_numeric(montant, errors='coerce')

    # apply the conversions
    montant = montant.where(~is_usd, montant * 130)
    montant = montant.where(~is_k, montant * 100)

    print(montant.dtypes)
    print(montant)


# Fix type
...

if __name__ == '__main__':
    import os

    print(os.listdir(os.getcwd()))  # listing files
    step('Chargement des donnees')
    df = load_data()
    print(df.head(10).to_string())
    print('data loaded successfully')
    step('Informations')
    df.info()

    df = remove_columns(df)
    print('DF revalidated')
    df.info()
    X, y = build_X_y(df)
    print('Target variable', y)

    show(X, 10)

    step('Fixing dates')
    fix_dates(X)

    show(X, 20)

    fix_montant(X)
