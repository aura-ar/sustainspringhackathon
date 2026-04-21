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

y_maize = y_filter1[y_filter1["Item"] == "Maize"]

# x_maize = Pesticides.merge(
#     y_maize[["Area", "Year"]].drop_duplicates(),
#     on=["Area", "Year"],
#     how="inner"
# )

maize_data = Pesticides.merge(
    y_maize,
    on=["Area", "Year"],
    how="inner",
    suffixes=("_pest", "_yield")
)

X = maize_data["Value_pest"]
y = maize_data["Value_yield"]  


# print(maize_data.shape)
# print(maize_data.head(30))    
# print(y_maize.shape)
# print(x_maize.shape)
print(maize_data.columns.tolist())