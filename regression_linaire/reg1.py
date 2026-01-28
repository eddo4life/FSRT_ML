import numpy as np
import pandas as pd


# Chargement / Traitement des donnees
# ===================================

# Construction de Xy avec le biais
def construire_Xy_biais(nom_fichier):
    df = pd.read_excel(nom_fichier)

    # Creation de la matrice des features X et du vecteur cible y
    X = df[["Surface", "Anciennete"]].to_numpy(dtype=float)
    y = df["Prix_kUS"].to_numpy(dtype=float)

    # Ajout de la colonne de 1 pour le biais w0 a X
    m = X.shape[0]
    X = np.column_stack([np.ones(m), X])

    return X, y, df


# Le Modele: la regression lineaire
# =========================

# Calcul du cout
def fonction_cout_J(X, y, w):
    m = X.shape[0]
    r = X @ w - y  # le vecteur des redidus
    return (1 / (2 * m)) * np.linalg.norm(r) ** 2


# descente de gradient
def entrainer_reg_lin_dg(X, y, learning_rate, n_iters=5000, mode_verbeux=True):
    m, n = X.shape

    # initialisons les wj a 0.0
    w = np.zeros(n, dtype=float)

    for it in range(1, n_iters + 1):
        y_chap = X @ w
        err = y_chap - y

        # Cout
        J = (1 / (2 * m)) * np.sum(err ** 2)

        # Gradient
        grad = (X.T @ err) / m

        # Mise a jour le vecteur w
        w = w - learning_rate * grad

        if mode_verbeux and (it == 1 or it % 5000 == 0 or it == n_iters):
            print(f"Iter {it:5d} | J(w) = {J:.6f} | w = {w}")

    return w


# Calcul d'une prediction pour une donnee
def prediction(surface, anciennete, w):
    x = np.array([1.0, float(surface), float(anciennete)])
    return float(x @ w)


# Evaluationdu modele
def metriques_regression(y_reel, y_pred):
    y_true = np.asarray(y_rell, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_reel - y_pred) ** 2)
    ss_tot = np.sum((y_reel - np.mean(y_reel)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else float("nan")

    rmse = np.sqrt(np.mean((y_reel - y_pred) ** 2))
    mae = np.mean(np.abs(y_reel - y_pred))

    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def afficher_performance(X, y, w):
    y_pred = X @ w
    mets = metriques_regression(y, y_pred)
    J = fonction_cout_J(X, y, w)

    print("\n=== Performance sur le dataset charge ===")
    print(f"J(w)  = {J:.6f}")
    print(f"R^2    = {mets['R2']:.6f}")
    print(f"RMSE  = {mets['RMSE']:.6f}")
    print(f"MAE   = {mets['MAE']:.6f}")


def afficher_parametres(w):
    print("\nParametres appris :")
    print(f"w0 (biais)          = {w[0]:.6f}")
    print(f"w1 (Surface_m2)     = {w[1]:.6f}")
    print(f"w2 (Anciennete_ans) = {w[2]:.6f}")


# Menu
def afficher_menu():
    print("\n==============================")
    print("  Regression Lineaire (DG)")
    print("==============================")
    print("1) Charger jeu de donnees")
    print("2) Entrainer modele")
    print("3) Predire une valeur")
    print("4) Afficher performance du modele (R2, RMSE, MAE, J)")
    print("5) Afficher parametres du modele")
    print("0) Quitter")


# Utilitaires de saisie
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


# Fonction principale
def main():
    # Initialisation
    X = None
    y = None
    df = None
    w = None

    # Parametres d'entrainement par defaut
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
                X, y, df = construire_Xy_biais(nom)
                w = None  # on invalide le modèle si on recharge un dataset
                print(f"Dataset charge: {nom}")
                print(f"Nombre d'exemples: {X.shape[0]}, Nombre de features (avec biais): {X.shape[1]}")
                print("Aperçu:")
                print(df.head())
            except Exception as e:
                print(f"Erreur de chargement: {e}")

        elif choix == "2":
            if X is None or y is None:
                print("Charger d'abord un jeu de donnees (option 1).")
                continue

            lr = lire_float("Learning rate (ex: 0.0001): ")
            n_iters = lire_int(f"Nombre d'iterations (defaut {n_iters_defaut}): ")
            if n_iters <= 0:
                print("Le nombre d'iterations doit être > 0.")
                continue

            w = entrainer_reg_lin_dg(X, y, lr, n_iters, mode_verbeux=True)
            print("\nEntrainement termine.")
            afficher_parametres(w)

        elif choix == "3":
            if w is None:
                print("Veuillez d'abord entrainer le modele (option 2).")
                continue

            surface = lire_float("Surface (m2): ")
            anciennete = lire_float("Anciennete (ans): ")
            pred = prediction(surface, anciennete, w)

            print("\n=== Prediction ===")
            print(f"Surface = {surface} m2, Anciennete = {anciennete} ans")
            print(f"Prix predit = {pred:.2f} kUS")

        elif choix == "4":
            if X is None or y is None:
                print("Veuillez d'abord charger un jeu de donnees (option 1).")
                continue
            if w is None:
                print("Veuillez d'abord entraîner le modele (option 2).")
                continue

            afficher_performance(X, y, w)

        elif choix == "5":
            if w is None:
                print("Modele non entraîne. Option 2 d'abord.")
                continue
            afficher_parametres(w)

        else:
            print("Option invalide. Reessayez.")


if __name__ == "__main__":
    import os

    print(os.listdir(os.getcwd()))
    main()
