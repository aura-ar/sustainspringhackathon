import pandas as pd

# # Load the data
Pesticides = pd.read_csv("datasets/pesticides.csv")
Yield = pd.read_csv("datasets/yield.csv")
Temperature = pd.read_csv("datasets/temp.csv")
Rainfall = pd.read_csv("datasets/rainfall.csv")

# # Filter yield to match pesticides
# y_filter1= Yield.merge(
#     Pesticides[["Area", "Year"]].drop_duplicates(),
#     on=["Area", "Year"],
#     how="inner"
# )
Yield.columns = Yield.columns.str.strip()
Pesticides.columns = Pesticides.columns.str.strip()
Rainfall.columns = Rainfall.columns.str.strip()
Temperature.columns = Temperature.columns.str.strip()

Temperature = Temperature.rename(columns={
    "country": "Area",
    "year": "Year",
    "avg_temp": "avg_temp"
})

Rainfall = Rainfall.rename(columns={
    "Area": "Area",
    "Year": "Year", 
    "average_rain_fall_mm_per_year": "Average_Rainfall"
})

Yield = Yield.rename(columns={"Value": "Value_yield"})

Temperaturena = Temperature.copy()
Temperaturena["avg_temp"] = Temperature["avg_temp"].astype(str).str.strip().replace("", pd.NA).dropna().astype(float)
Rainfallna = Rainfall.copy()
Rainfallna["Average_Rainfall"] = Rainfall["Average_Rainfall"].astype(str).str.strip().replace("", pd.NA).dropna()
TemperatureFinal = Temperaturena.groupby(["Area", "Year"], as_index=False)["avg_temp"].mean()
rain_temp1 = TemperatureFinal.merge(Rainfall[["Area", "Year"]].drop_duplicates(),on=["Area", "Year"],how="inner")
raintempfinal = Rainfall.merge(rain_temp1,on=["Area", "Year"],how="inner", suffixes=("_temp", "_rain") )


print(raintempfinal.shape)


# # y_maize = y_filter1[y_filter1["Item"] == "Maize"]

# # x_maize = Pesticides.merge(
# #     y_maize[["Area", "Year"]].drop_duplicates(),
# #     on=["Area", "Year"],
# #     how="inner"
# # )

# data = Pesticides.merge(
#         y_filter1,
#         on=["Area", "Year"],
#         how="inner",
#         suffixes=("_pest", "_yield")
#     )

# # X = data[["Value_pest", "Year", "Item"]]
# # y = data["Value_yield"]  


# print(data.shape)
# print(data.head(30))    
# # print(y_maize.shape)
# # print(x_maize.shape)
# # print(X.columns.tolist())