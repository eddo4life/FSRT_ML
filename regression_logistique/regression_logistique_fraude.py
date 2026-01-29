import numpy as np
import pandas as pd


# Chargement / Traitement des donnees
# ===================================

# Construction de Xy avec le biais w0 + normalisation (standardisation) des features
def construire_Xy_biais(nom_fichier, standardiser=True):
    df = pd.read_excel(nom_fichier)

    print(df)

    # Features (sans biais) et cible
    X0 = df[["Montant_USD", "NbTrans_24h", "AncienneteCompte_j"]].to_numpy(dtype=float)
    y = df["Fraude"].to_numpy(dtype=float)

    # Parametres de normalisation (pour re-normaliser en prediction)
    mu = None
    sigma = None

    if standardiser:
        mu = X0.mean(axis=0)
        sigma = X0.std(axis=0)
        sigma[sigma == 0] = 1.0  # evite division par zero
        X0 = (X0 - mu) / sigma

    # Ajout biais w0
    m = X0.shape[0]
    X = np.column_stack([np.ones(m), X0])

    return X, y, df, mu, sigma


# Le Modele: la regression logistique binaire
# ===========================================

# Sigmoide simple + protection numerique
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# Cout log-loss
def fonction_cout_J(X, y, w):
    m = X.shape[0]
    p = sigmoid(X @ w)

    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    return (1.0 / m) * np.sum(-y * np.log(p) - (1 - y) * np.log(1 - p))


# Descente de gradient
def entrainer_reg_log_dg(X, y, learning_rate, n_iters=5000, mode_verbeux=True):

    m, n = X.shape
    w = np.zeros(n, dtype=float)

    for it in range(1, n_iters + 1):
        p = sigmoid(X @ w)
        err = p - y

        J = fonction_cout_J(X, y, w)
        grad = (X.T @ err) / m
        w = w - learning_rate * grad

        if mode_verbeux and (it == 1 or it % 1000 == 0 or it == n_iters):
            print(f"Iter {it:5d} | J(w) = {J:.6f} | w = {w}")

    return w


# Prediction: probabilite et classe (avec renormalisation si mu/sigma fournis)
def proba_prediction(montant_usd, nbtrans_24h, anciennete_compte_j, w, mu=None, sigma=None):
    x = np.array([float(montant_usd), float(nbtrans_24h), float(anciennete_compte_j)], dtype=float)

    if mu is not None and sigma is not None:
        x = (x - mu) / sigma

    x = np.concatenate(([1.0], x))  # biais
    return float(sigmoid(x @ w))


def prediction_classe(montant_usd, nbtrans_24h, anciennete_compte_j, w, seuil=0.5, mu=None, sigma=None):
    p = proba_prediction(montant_usd, nbtrans_24h, anciennete_compte_j, w, mu=mu, sigma=sigma)
    return int(p >= seuil), p


# Metriques
def metriques_classification(y_reel, y_pred):
    y_reel = np.asarray(y_reel, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_reel == 1) & (y_pred == 1)))
    tn = int(np.sum((y_reel == 0) & (y_pred == 0)))
    fp = int(np.sum((y_reel == 0) & (y_pred == 1)))
    fn = int(np.sum((y_reel == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


def afficher_performance(X, y, w, seuil=0.5):
    p = sigmoid(X @ w)
    y_pred = (p >= seuil).astype(int)

    mets = metriques_classification(y, y_pred)
    J = fonction_cout_J(X, y, w)

    print("\n=== Performance sur le dataset charge ===")
    print(f"J(w)      = {J:.6f}")
    print(f"Accuracy  = {mets['Accuracy']:.6f}")
    print(f"Precision = {mets['Precision']:.6f}")
    print(f"Recall    = {mets['Recall']:.6f}")
    print(f"F1        = {mets['F1']:.6f}")
    print(f"Confusion: TP={mets['TP']} | TN={mets['TN']} | FP={mets['FP']} | FN={mets['FN']}")


def afficher_parametres(w):
    print("\nParametres appris :")
    print(f"w0 (biais)                = {w[0]:.6f}")
    print(f"w1 (Montant_USD)          = {w[1]:.6f}")
    print(f"w2 (NbTrans_24h)          = {w[2]:.6f}")
    print(f"w3 (AncienneteCompte_j)   = {w[3]:.6f}")


# Menu
def afficher_menu():
    print("\n====================================")
    print("  Regression Logistique Binaire (DG)")
    print("        Detection de fraude")
    print("====================================")
    print("1) Charger jeu de donnees (avec normalisation)")
    print("2) Entrainer modele")
    print("3) Predire (classe + probabilite)")
    print("4) Afficher performance du modele (Acc, Prec, Recall, F1, J)")
    print("5) Afficher parametres du modele")
    print("0) Quitter")


def lire_float(prompt):
    while True:
        s = input(prompt).strip().replace(",", ".")
        try:
            return float(s)
        except ValueError:
            print("Entree invalide. Merci de saisir un nombre.")


def lire_int(prompt):
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("Entree invalide. Merci de saisir un entier.")


def main():
    X = None
    y = None
    df = None
    w = None
    mu = None
    sigma = None

    n_iters_defaut = 5000

    while True:
        afficher_menu()
        choix = input("Choisissez une option: ").strip()

        if choix == "0":
            print("Fin du programme.")
            break

        elif choix == "1":
            nom = input("Nom et chemin du fichier Excel: ").strip()
            try:
                X, y, df, mu, sigma = construire_Xy_biais(nom, standardiser=True)
                w = None
                print(f"Dataset charge: {nom}")
                print(f"Nombre d'exemples: {X.shape[0]}, Features (avec biais): {X.shape[1]}")
                print("Aperçu:")
                print(df.head())
            except Exception as e:
                print(f"Erreur de chargement: {e}")

        elif choix == "2":
            if X is None or y is None:
                print("Charger d'abord un jeu de donnees (option 1).")
                continue

            lr = lire_float("Learning rate (ex: 0.05): ")
            n_iters = lire_int(f"Nombre d'iterations (defaut {n_iters_defaut}): ")
            if n_iters <= 0:
                print("Le nombre d'iterations doit etre > 0.")
                continue

            w = entrainer_reg_log_dg(X, y, lr, n_iters, mode_verbeux=True)
            print("\nEntrainement termine.")
            afficher_parametres(w)

        elif choix == "3":
            if w is None:
                print("Veuillez d'abord entrainer le modele (option 2).")
                continue

            montant = lire_float("Montant_USD: ")
            nbtrans = lire_float("NbTrans_24h: ")
            anciennete = lire_float("AncienneteCompte_j: ")

            y_hat, p = prediction_classe(montant, nbtrans, anciennete, w, seuil=0.5, mu=mu, sigma=sigma)

            print("\n=== Prediction ===")
            print(f"P(Fraude=1 | x) = {p:.4f}")
            print(f"Classe predite (seuil 0.5) = {y_hat}")

        elif choix == "4":
            if X is None or y is None:
                print("Veuillez d'abord charger un jeu de donnees (option 1).")
                continue
            if w is None:
                print("Veuillez d'abord entrainer le modele (option 2).")
                continue

            afficher_performance(X, y, w, seuil=0.5)

        elif choix == "5":
            if w is None:
                print("Modele non entraine. Option 2 d'abord.")
                continue
            afficher_parametres(w)

        else:
            print("Option invalide. Reessayez.")


if __name__ == "__main__":
    import os

    print(os.listdir(os.getcwd()))
    main()
