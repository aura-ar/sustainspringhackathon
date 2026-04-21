import pandas as pd

# Load the data
Pesticides = pd.read_csv("datasets/pesticides.csv")
Yield = pd.read_csv("datasets/yield.csv")


# Filter yield to match pesticides
y_filter1= Yield.merge(
    Pesticides[["Area", "Year"]].drop_duplicates(),
    on=["Area", "Year"],
    how="inner"
)

# y_maize = y_filter1[y_filter1["Item"] == "Maize"]

# x_maize = Pesticides.merge(
#     y_maize[["Area", "Year"]].drop_duplicates(),
#     on=["Area", "Year"],
#     how="inner"
# )

data = Pesticides.merge(
        y_filter1,
        on=["Area", "Year"],
        how="inner",
        suffixes=("_pest", "_yield")
    )

# X = data[["Value_pest", "Year", "Item"]]
# y = data["Value_yield"]  


print(data.shape)
print(data.head(30))    
# print(y_maize.shape)
# print(x_maize.shape)
# print(X.columns.tolist())