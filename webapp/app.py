from flask import Flask, jsonify, render_template, request
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

def build_features(dataframe):
    return pd.get_dummies(dataframe[["Value_pest", "Year", "Item_yield"]])

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

    # You can probably just copy most of the code and change the application to answer the other questions we want to do. The second merge is important to make sure that they are lined up correctly.
    else:
        pass

    # mode = request.args.get("mode")
    # if mode == "Maize":

    #     y_maize = y_filter1[y_filter1["Item"] == "Maize"]

    #     # x_maize = Pesticides.merge(
    #     #     y_maize[["Area", "Year"]].drop_duplicates(),
    #     #     on=["Area", "Year"],
    #     #     how="inner"
    #     # )

    #     maize_data = Pesticides.merge(
    #         y_maize,
    #         on=["Area", "Year"],
    #         how="inner",
    #         suffixes=("_pest", "_yield")
    #     )

    #     X = build_features(maize_data)
    #     y = maize_data["Value_yield"]  
    #     # X = maize_data
    #     # y = maize_data

    # elif mode == "Wheat":
    #     y_wheat = y_filter1[y_filter1["Item"] == "Wheat"]

    #     wheat_data = Pesticides.merge(
    #         y_wheat,
    #         on=["Area", "Year"],
    #         how="inner",
    #         suffixes=("_pest", "_yield")
    #     )

    #     X = build_features(wheat_data)
    #     y = wheat_data["Value_yield"]

    # elif mode == "Potato":
    #     y_potato = y_filter1[y_filter1["Item"] == "Potatoes"]

    #     potato_data = Pesticides.merge(
    #         y_potato,
    #         on=["Area", "Year"],
    #         how="inner",
    #         suffixes=("_pest", "_yield")
    #     )

    #     X = build_features(potato_data)
    #     y = potato_data["Value_yield"]
                               

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


# Start the server if this file is run directly (python app.py)
if __name__ == "__main__":
    # host="0.0.0.0":
    # listen on all network interfaces without it Flask only listens inside the container 
    # and your browser can't reach it.
    app.run(host="0.0.0.0", port=5000, debug=True)
