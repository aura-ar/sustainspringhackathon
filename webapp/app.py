from flask import Flask, jsonify, render_template, request
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

def build_features(dataframe):
    return pd.get_dummies(dataframe[["Value_pest", "Year", "Item_yield"]])

def build_features2(dataframe):
    return pd.get_dummies(dataframe[["avg_temp", "Year"]])

def build_features3(dataframe):
    return pd.get_dummies(dataframe[["avg_temp", "Year", "Predicted_Rainfall"]])

# This route "/" is the default page, this will run when the browerser runs the URL
@app.route("/")
def index():
    return render_template("index.html")

# This is called by a acript in the index.html
@app.route("/run")
def run_model():

    
    # Read the seed from the query string e.g. /run?seed=42, default to 42 if not provided
    seed = int(request.args.get("seed", 42))

    #Data preprocessing

    Pesticides = pd.read_csv("datasets/pesticides.csv")
    Yield = pd.read_csv("datasets/yield.csv")
    Rainfall = pd.read_csv("datasets/rainfall.csv")
    Temperature = pd.read_csv("datasets/temp.csv")

    mode = request.args.get("mode")
    if mode == "Pest":

        # Filter yield to match pesticides
        y_filter1= Yield.merge(
            Pesticides[["Area", "Year"]].drop_duplicates(),
            on=["Area", "Year"],
            how="inner"
        )

        data = Pesticides.merge(
            y_filter1,
            on=["Area", "Year"],
            how="inner",
            suffixes=("_pest", "_yield")
        )

        X = build_features(data)
        y = data["Value_yield"]  

        #split into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        #Create, fit and run out model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # results /  actuals
        y_test_values = y_test.to_numpy()
        results = [
            {"index": i, "actual": round(float(y_test_values[i]), 2), "predicted": round(float(preds[i]), 2)}
            for i in range(min(50, len(y_test_values)))
        ]
        #feature importance results
        importances = [
            {"feature": feature_name, "importance": round(float(importance), 4)}
            for feature_name, importance in zip(X.columns, model.feature_importances_)
        ]
        #return these results to script in index.html
        return jsonify({"results": results, "importances": importances})

    # You can probably just copy most of the code and change the application to answer the other questions we want to do. The second merge is important to make sure that they are lined up correctly.
    elif mode == "Rain":

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

        TemperatureFinal = Temperature.groupby(["Area", "Year"], as_index=False)["avg_temp"].mean()
        rain_temp1 = TemperatureFinal.merge(Rainfall[["Area", "Year"]].drop_duplicates(),on=["Area", "Year"],how="inner")
        raintempfinal = Rainfall.merge(rain_temp1,on=["Area", "Year"],how="inner", suffixes=("_temp", "_rain") )

        # Filter out rows with invalid/empty values in numeric columns
        raintempfinal['avg_temp'] = pd.to_numeric(raintempfinal['avg_temp'], errors='coerce')
        raintempfinal['Average_Rainfall'] = pd.to_numeric(raintempfinal['Average_Rainfall'], errors='coerce')
        raintempfinal = raintempfinal.dropna(subset=['avg_temp', 'Average_Rainfall'])

        X= build_features2(raintempfinal)
        y = raintempfinal["Average_Rainfall"]

        #split into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        #Create, fit and run out model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # results /  actuals
        y_test_values = y_test.to_numpy()
        results = [
            {"index": i, "actual": round(float(y_test_values[i]), 2), "predicted": round(float(preds[i]), 2)}
            for i in range(min(50, len(y_test_values)))
        ]
        #feature importance results
        importances = [
            {"feature": feature_name, "importance": round(float(importance), 4)}
            for feature_name, importance in zip(X.columns, model.feature_importances_)
        ]
        #return these results to script in index.html
        return jsonify({"results": results, "importances": importances})

    elif mode == "RainYield":

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

        TemperatureFinal = Temperature.groupby(["Area", "Year"], as_index=False)["avg_temp"].mean()
        rain_temp1 = TemperatureFinal.merge(Rainfall[["Area", "Year"]].drop_duplicates(),on=["Area", "Year"],how="inner")
        raintempfinal = Rainfall.merge(rain_temp1,on=["Area", "Year"],how="inner", suffixes=("_temp", "_rain") )

        # Filter out rows with invalid/empty values in numeric columns
        raintempfinal['avg_temp'] = pd.to_numeric(raintempfinal['avg_temp'], errors='coerce')
        raintempfinal['Average_Rainfall'] = pd.to_numeric(raintempfinal['Average_Rainfall'], errors='coerce')
        raintempfinal = raintempfinal.dropna(subset=['avg_temp', 'Average_Rainfall'])

        Yield['Value_yield'] = pd.to_numeric(Yield['Value_yield'], errors='coerce')
        raintempyield = raintempfinal.merge(
            Yield[["Area", "Year", "Value_yield"]],
            on=["Area", "Year"],
            how="inner"
        )
        raintempyield = raintempyield.dropna(subset=['avg_temp', 'Average_Rainfall', 'Value_yield'])

        rain_features = build_features2(raintempyield)
        rain_target = raintempyield["Average_Rainfall"]
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        predicted_rainfall = np.zeros(len(raintempyield))

        for train_idx, val_idx in kf.split(rain_features):
            X_tr, X_val = rain_features.iloc[train_idx], rain_features.iloc[val_idx]
            y_tr = rain_target.iloc[train_idx]

            rain_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
            rain_model.fit(X_tr, y_tr)
            predicted_rainfall[val_idx] = rain_model.predict(X_val)

        raintempyield["Predicted_Rainfall"] = predicted_rainfall

        X = build_features3(raintempyield)
        y = raintempyield["Value_yield"]

        #split into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        #Create, fit and run out model
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # results /  actuals
        y_test_values = y_test.to_numpy()
        results = [
            {"index": i, "actual": round(float(y_test_values[i]), 2), "predicted": round(float(preds[i]), 2)}
            for i in range(min(50, len(y_test_values)))
        ]
        #feature importance results
        importances = [
            {"feature": feature_name, "importance": round(float(importance), 4)}
            for feature_name, importance in zip(X.columns, model.feature_importances_)
        ]
        #return these results to script in index.html
        return jsonify({"results": results, "importances": importances})
    
    elif mode == "Country":

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

        data = Yield.merge(
            Pesticides[["Area", "Year", "Value"]].rename(columns={"Value": "Value_pest"}), # Rename Pesticides' Value here
            on=["Area", "Year"],
            how="left"
        )

        data = data.merge(
            Rainfall[["Area", "Year", "Average_Rainfall"]],
            on=["Area", "Year"],
            how="left"
        )

        data = data.merge(
            Temperature[["Area", "Year", "avg_temp"]],
            on=["Area", "Year"],
            how="left"
        )

        data = data.dropna(subset=["Value_yield", "Area", "Year"])

        results = []
        importances = []
        country_details = []

        for country_x in ["India", "Brazil", "Mexico", "Pakistan", "Canada"]:
            train_data = data[data["Year"] <= 2008].copy()
            test_data = data[data["Year"] > 2008].copy()

            global_train = train_data.copy()
            global_test = test_data[test_data["Area"] == country_x].copy()

            local_train = train_data[train_data["Area"] == country_x].copy()
            local_test = test_data[test_data["Area"] == country_x].copy()

            if global_test.empty or local_train.empty or local_test.empty:
                continue

            feature_cols = ["Value_pest", "Year", "Item", "Area", "Average_Rainfall", "avg_temp"]

            X_global_train = pd.get_dummies(global_train[feature_cols])
            X_global_test = pd.get_dummies(global_test[feature_cols])
            X_global_test = X_global_test.reindex(columns=X_global_train.columns, fill_value=0)

            y_global_train = global_train["Value_yield"]
            y_global_test = global_test["Value_yield"]

            global_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
            global_model.fit(X_global_train, y_global_train)
            global_preds = global_model.predict(X_global_test)

            X_local_train = pd.get_dummies(local_train[feature_cols])
            X_local_test = pd.get_dummies(local_test[feature_cols])
            X_local_test = X_local_test.reindex(columns=X_local_train.columns, fill_value=0)

            y_local_train = local_train["Value_yield"]
            y_local_test = local_test["Value_yield"]

            local_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=42
            )
            local_model.fit(X_local_train, y_local_train)
            local_preds = local_model.predict(X_local_test)

            global_mae = float((abs(y_global_test.to_numpy() - global_preds)).mean())
            local_mae = float((abs(y_local_test.to_numpy() - local_preds)).mean())

            results.append({
                "index": country_x,
                "actual": round(global_mae, 2),
                "predicted": round(local_mae, 2)
            })
            importances.append({
                "feature": country_x,
                "importance": round(abs(global_mae - local_mae), 2)
            })
            country_details.append({
                "country": country_x,
                "global_mae": round(global_mae, 2),
                "local_mae": round(local_mae, 2),
                "winner": "Local model" if local_mae < global_mae else "Global model"
            })

        return jsonify({
            "results": results,
            "importances": importances,
            "country_results": country_details
        })

    elif mode == "CountryByModel":
        Pesticides.columns = Pesticides.columns.str.strip()
        Yield.columns = Yield.columns.str.strip()
        Rainfall.columns = Rainfall.columns.str.strip()
        Temperature.columns = Temperature.columns.str.strip()

        # Standardise names
        Temperature = Temperature.rename(columns={"country": "Area", "year": "Year"})
        Rainfall = Rainfall.rename(columns={"average_rain_fall_mm_per_year": "Average_Rainfall"})
        Yield = Yield.rename(columns={"Value": "Value_yield"})

        # Merge all data
        data = Yield.merge(
            Pesticides[["Area", "Year", "Value"]],
            on=["Area", "Year"],
            how="left"
        ).rename(columns={"Value": "Value_pest"})

        data = data.merge(
            Rainfall[["Area", "Year", "Average_Rainfall"]],
            on=["Area", "Year"],
            how="left"
        )

        data = data.merge(
            Temperature[["Area", "Year", "avg_temp"]],
            on=["Area", "Year"],
            how="left"
        )

        data = data.dropna(subset=[
            "Value_yield", "Area", "Year", "Item",
            "Value_pest", "Average_Rainfall", "avg_temp"
        ])

        country_x = "Canada"
        split_year = 2008

        train_data = data[data["Year"] <= split_year].copy()
        test_data = data[data["Year"] > split_year].copy()

        global_train = train_data.copy()
        global_test = test_data[test_data["Area"] == country_x].copy()

        local_train = train_data[train_data["Area"] == country_x].copy()
        local_test = test_data[test_data["Area"] == country_x].copy()
        if len(global_test) == 0 or len(local_train) == 0 or len(local_test) == 0:
            raise ValueError(f"Not enough data for {country_x}")

        feature_cols = ["Value_pest", "Year", "Item", "Area", "Average_Rainfall", "avg_temp"]
        # Build encoded datasets once
        X_global_train = pd.get_dummies(global_train[feature_cols])
        X_global_test = pd.get_dummies(global_test[feature_cols])
        X_global_test = X_global_test.reindex(columns=X_global_train.columns, fill_value=0)

        X_local_train = pd.get_dummies(local_train[feature_cols])
        X_local_test = pd.get_dummies(local_test[feature_cols])
        X_local_test = X_local_test.reindex(columns=X_local_train.columns, fill_value=0)

        y_global_train = global_train["Value_yield"]
        y_global_test = global_test["Value_yield"]

        y_local_train = local_train["Value_yield"]
        y_local_test = local_test["Value_yield"]

        results = []

        def compare_model(model, model_name):
            global_model = model
            global_model.fit(X_global_train, y_global_train)
            global_preds = global_model.predict(X_global_test)
            global_mae = mean_absolute_error(y_global_test, global_preds)

            # create a fresh model of the same type for local training
            if model_name == "XGBoost":
                local_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    random_state=42
                )
            else:
                local_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

            local_model.fit(X_local_train, y_local_train)
            local_preds = local_model.predict(X_local_test)
            local_mae = mean_absolute_error(y_local_test, local_preds)

            results.append({
                "Model": model_name,
                "Country": country_x,
                "Global MAE": round(global_mae, 2),
                "Local MAE": round(local_mae, 2),
                "Winner": "Local model" if local_mae < global_mae else "Global model"
            })

        compare_model(
            xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42),
            "XGBoost"
        )

        compare_model(
            RandomForestRegressor(n_estimators=100, random_state=42),
            "Random Forest"
        )

        # "actual" = global model MAE, "predicted" = local model MAE
        # This maps onto the existing frontend line chart keys
        chart_data = [
            {
                "index": r["Model"],
                "actual": r["Global MAE"],
                "predicted": r["Local MAE"]
            }
            for r in results
        ]
        # Bar chart shows how much the local model differs from the global model
        importances = [
            {"feature": r["Model"], "importance": round(abs(r["Global MAE"] - r["Local MAE"]), 2)}
            for r in results
        ]
        return jsonify({"results": chart_data, "importances": importances, "country_results": results})


# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
