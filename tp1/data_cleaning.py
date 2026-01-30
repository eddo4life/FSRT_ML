import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Just to identify different steps
def step(msg=''):
    dash = 40
    print(f'[{msg}]','-'*dash,sep='')
    print('-'*(dash+len(msg)+2))

#load data
def load_data():
    print('loading data')
    try:
        return pd.read_excel('dataset_pretraitement_fraude.xlsx')
    except Exception as e:
        print(f'Erreur de chargement {e}')
        sys.exit(0) #TODO add an interactive mode if I got enough time, hopefully :)

def revalidated_df(df):
    # On enleve TransactionID, ClientID, Commentaire
    print('revalidating df')
    return df.drop(columns=['TransactionID', 'ClientID', 'Commentaire'])



if __name__ == '__main__':
    import os
    print(os.listdir(os.getcwd())) #listing files
    step('Chargement des donnees')
    df = load_data()
    print(df.head())
    print('data loaded successfully')

    df = revalidated_df(df)
    print('DF revalidated')

