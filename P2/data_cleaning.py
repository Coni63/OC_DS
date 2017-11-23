import pandas as pd

def custom_split(x):
    """
    Retourne le nb d'elements d'une string séparé par une virgule
    :param x: une string séparés par des ","
    :return: le nb de sequences
    """
    if pd.isnull(x):
        return 0
    else:
        return len(x.split(','))

def cleanup(word):
    """
    Certains labels n'ont pas le m^me format que d'autres (spérés par des tirets et pas de majuscule au depart). Cette fonction corrige cette erreur pour fusionner les lables
    :param word: la catégorie mal écrite
    :return: la catégorie bien écrite
    """
    if not pd.isnull(word):
        if word.count("-") > 0:
            word = word.replace("-", " ")
            word = word[0].upper() + word[1:]
    return word

def cleanup2(word):
    """
        Certains labels n'ont pas le même format que d'autres pour pnns2.
        :param word: la catégorie mal écrite
        :return: la catégorie bien écrite
    """
    if not pd.isnull(word):
        return word.title()

# Chargement des données
print("opening dataset")
df = pd.read_csv("dataset/fr.openfoodfacts.org.products.csv", sep='\t', dtype=object)

# suppression des features toujours vide
print("remove empty features")
df.dropna(axis=1, how='all', inplace=True)

# suppresion des lignes dont les informations nutritionnelles sont manquantes
subset = [col for col in df if col.endswith("_100g")]
df.dropna(subset=subset[:-3], how='all', inplace=True)

# conversion des labels
print("convert/aggregate features")
df["labels"] = df["labels"].apply(custom_split)
df["traces"] = df["traces"].apply(custom_split)
df["allergens"] = df["allergens"].apply(custom_split)
df["pnns_groups_1"] = df["pnns_groups_1"].apply(cleanup)
df["pnns_groups_2"] = df["pnns_groups_2"].apply(cleanup2)

# suppresions des labels "inutiles"
print("removing features")
subset = [  "code", "url", "creator", "quantity", "carbon-footprint_100g",
            "packaging", "packaging_tags",
            "brands", "brands_tags",
            "categories", "categories_tags", "categories_fr",
            "origins", "origins_tags",
            "manufacturing_places", "manufacturing_places_tags",
            "labels_tags", "labels_fr",
            "emb_codes", "emb_codes_tags",
            "first_packaging_code_geo",
            "cities", "cities_tags",
            "purchase_places", "stores",
            "countries", "countries_fr", "countries_tags",
            "ingredients_text", "traces_tags", "traces_fr",                                  # on a le compte
            "ingredients_from_palm_oil_tags", "ingredients_that_may_be_from_palm_oil_tags",  # on a le compte
            "additives", "additives_tags", "additives_fr",          # on a additives_n
            "serving_size",                                         # on a les données /100g, c'est ingérable par portion avec trop de données manquantes
            "main_category", "main_category_fr",                    # suppression car on a pnns_groups_1/2
            "image_small_url", "image_url",
            "states", "states_tags", "states_fr",
            "allergens_fr",
            "nutrition-score-fr_100g",                              # on reste sur le score FSA
            "sodium_100g"                                           # sodium_100g = salt_100g
        ]
df.drop(labels=subset, axis=1,inplace=True)

# suppressions des données temporelles inutiles
for column in df:
    if column.endswith("_datetime") or column.endswith("_t") or column.endswith("_name"):
        df.drop(labels=[column], axis=1,inplace=True)

# Conversion des colonnes
for column in df:
    if column.endswith("_100g") or column.endswith("_serving"):
        df[column] = df[column].astype("float")

# Nettoyages des outliers
print("remove outliers")
df = df[(df["energy_100g"] < 4000) | (df["energy_100g"].isnull())]   # Valeur basée sur le boxplot

# Basé sur les boxplot, on a difficilement plus de 2g/100g de vitamine
# On ne peut pas avoir plus de 100g de certaines informations nutritionnelles par 100g ou une valeur negative
for column in df:
    if column.startswith("vitamin"):
        df = df[(df[column] < 2) | (df[column].isnull())]
    if column.endswith("_100g") and not column.startswith(("energy", "nutrition-score")):
        df = df[((df[column] >= 0) & (df[column] < 100)) | (df[column].isnull())]

# d'autres mineraux sont aussi sous les 2%
minerals=["trans-fat_100g", "pantothenic-acid_100g", "potassium_100g", "calcium_100g", "phosphorus_100g", "iron_100g"]
for each in minerals:
    df = df[(df[each] < 2) | (df[each].isnull())]

# custom_scale
# Malgré les seuil a 2g, certaines features parraissent très inégales.
# du coup j'ai tenter de limiter le scale a la moyenne + 5 * la deviation (mais sans succes probant)
threshold = 5
for column in df:
    if column.endswith("_100g") and not column.startswith(("energy", "nutrition-score")):
        max_valid = df[column].median() + threshold * df[column].std()
        df = df[(df[column] <= max_valid) | (df[column].isnull())]
        df = df[(df[column] > 0) | (df[column].isnull())]


# Les graisses particulières ne peuvent pas etre suppérieur aux graisses totales
print("removing fake points")
df = df[(df["monounsaturated-fat_100g"] < df["fat_100g"])  | (df["monounsaturated-fat_100g"].isnull()) ]
df = df[(df["polyunsaturated-fat_100g"] < df["fat_100g"])  | (df["polyunsaturated-fat_100g"].isnull()) ]

# Après cleaning, suppression des features qui ont moins de 1000pts de données
threshold = 1000
for col in df :
    if col.endswith("_100g"):
        if df[col].notnull().sum() < threshold:
            df.drop(labels=[col], axis=1,inplace=True)

# Sauvegarde du dataset
print("saving clean dataset")
df.to_csv("dataset/cleaned_dataset.csv", sep='\t', index=False)