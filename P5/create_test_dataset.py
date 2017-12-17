import pandas as pd

# Ouverture du dataset
df = pd.read_excel("Online Retail.xlsx")

# Export d'un 1er dataset du cluster 1 (Client 12386.0)
df2 = df[df["CustomerID"] == 12386.0]
df2.to_csv("prod/test1.csv")

# Export d'un 1er dataset du cluster 2 (Client 18094.0)
df3 = df[df["CustomerID"] == 18094.0]
df3.to_csv("prod/test2.csv")

# Export d'un 1er dataset du cluster 3 (Client 17770.0)
df4 = df[df["CustomerID"] == 17770.0]
df4.to_csv("prod/test3.csv")

# Export d'un 1er dataset du cluster 4 (Client 12375.0)
df5 = df[df["CustomerID"] == 12375.0]
df5.to_csv("prod/test4.csv")

# Export d'un 1er dataset du cluster 5 (Client 18121.0)
df6 = df[df["CustomerID"] == 18121.0]
df6.to_csv("prod/test5.csv")

# Export d'un 1er dataset du cluster 7 (Client 12838.0)
df7 = df[df["CustomerID"] == 12838.0]
df7.to_csv("prod/test7.csv")

# Export d'un 1er dataset du cluster 8 (Client 18283.0)
df8 = df[df["CustomerID"] == 16923.0]
df8.to_csv("prod/test8.csv")