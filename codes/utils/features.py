import copy
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler

def feature_creation(df, math_creation=False, cat_group_creation=False):
    new_df = copy.deepcopy(df)

    # Math Features
    if(math_creation):
        numerical_columns = list(df.select_dtypes(include=[np.number]).columns.values)
        for i in range(0, len(numerical_columns)-1):
            col_1 = numerical_columns[i]
            new_df[f"{col_1}_pow2"] = df[col_1] ** 2
            new_df[f"{col_1}_pow3"] = df[col_1] ** 3
            new_df[f"{col_1}_mul10"] = df[col_1] * 10
            new_df[f"{col_1}_mul100"] = df[col_1] * 100
            for j in range(i+1, len(numerical_columns)):
                col_2 = numerical_columns[j]
                new_df[f"{col_1}_add_{col_2}"] = df[col_1] + df[col_2]
                new_df[f"{col_1}_sub_{col_2}"] = df[col_1] - df[col_2]
                new_df[f"{col_1}_subabs_{col_2}"] = np.abs(df[col_1] - df[col_2])
                new_df[f"{col_1}_mul_{col_2}"] = df[col_1] * df[col_2]
                new_df[f"{col_1}_truediv_{col_2}"] = np.where(df[col_2]!=0, df[col_1]/df[col_2], 0)
                new_df[f"{col_2}_truediv_{col_1}"] = np.where(df[col_1]!=0, df[col_2]/df[col_1], 0)
                new_df[f"{col_1}_floordiv_{col_2}"] = np.where(df[col_2]!=0, df[col_1]//df[col_2], 0)
                new_df[f"{col_2}_floordiv_{col_1}"] = np.where(df[col_1]!=0, df[col_2]//df[col_1], 0)

    # Categorical Features
    if(cat_group_creation):
        object_cat_columns = list(df.select_dtypes(include=['object']).columns.values)
        for i in range(0, len(object_cat_columns)-1):
            for j in range(i+1, len(object_cat_columns)):
                col_1 = object_cat_columns[i]
                col_2 = object_cat_columns[j]
                new_df[f"{col_1}_X_{col_2}"] = df[col_1] + "_" + df[col_2]

    return new_df

def extract_datetime_features(df, datetime_columns):
    date_df = pd.DataFrame()

    for col in datetime_columns:
        # Date
        date_df[f"{col}_Year"] = df[col].dt.year
        date_df[f"{col}_Quarter"] = df[col].dt.quarter
        date_df[f"{col}_Month"] = df[col].dt.month
        date_df[f"{col}_Day"] = df[col].dt.day
        date_df[f"{col}_Weekday"] = df[col].dt.weekday
        date_df[f"{col}_IsWeekend"] = (date_df[f"{col}_Weekday"] >= 5).astype(int)

        # Time
        date_df[f"{col}_Time"] = df[col].dt.time
        if(date_df[f"{col}_Time"].nunique()!=1):
            date_df[f"{col}_Hour"] = df[col].dt.hour
            date_df[f"{col}_Minute"] = df[col].dt.minute
            date_df[f"{col}_IsNight"] = ((date_df[f"{col}_Hour"] <= 6) | (date_df[f"{col}_Hour"] >= 18)).astype(int)
        date_df = date_df.drop(f"{col}_Time", axis=1)

    date_df = df.merge(date_df, left_index=True, right_index=True)
    return date_df

# Feature Engineering
def feature_engineering(df, config, fe_sets, split="Train", pipeline=None):
    # Splitting
    if(split=="Train"):
        pipeline = {}

    ids = df[config["id"]]
    df = df.drop(config["id"], axis=1)
    if("fold" in df.columns):
        df = df.drop("fold", axis=1)

    if(split!="Test"):
        X, y = df.drop(config["target"], axis=1), df[config["target"]]
        if(config["ml_task"]=="Classification" and len(y.unique())>2):
            if(split=="Train"):
                lb_encoder = LabelEncoder()
                y = lb_encoder.fit_transform(y)
                pipeline["Class Encoder"] = lb_encoder
            else:
                lb_encoder = pipeline["Class Encoder"]
                y = lb_encoder.transform(y)
    else:
        X = copy.deepcopy(df)

    # Feature Creation
    if(fe_sets["fe_feat_creation"]):
        X = feature_creation(X, fe_sets["fe_math_creation"], fe_sets["fe_cat_group_creation"])

    # Feature Extraction
    if(fe_sets["fe_feat_extraction"]):
        # Extract Datetime Features
        if(fe_sets["fe_datetime"]):
            if(split=="Train"):
                datetime_columns = list(X.select_dtypes(include=['datetime64[ns]']).columns.values)
                pipeline["Datetime Columns"] = datetime_columns
            else:
                datetime_columns = pipeline["Datetime Columns"]
            X = extract_datetime_features(X, datetime_columns)
            X = X.drop(datetime_columns, axis=1)

    # # Drop Features
    # if(len(drop_features) > 0):
    #     X = X.drop(drop_features, axis=1)

    # # Random Features
    # for i in range(3):
    #     X[f"rand_feat_{i+1}"] = np.random.rand(X.shape[0])

    # Filter Categorical Columns
    if(fe_sets["fe_drop_many_cat_unique"]):
        if(split=="Train"):
            categorical_columns = list(X.select_dtypes(include=['object', 'bool']).columns.values)
            many_categorical_columns = [col for col in categorical_columns if X[col].nunique() > fe_sets["fe_max_cat_unique"]]
            pipeline["Cat Columns with Many Unique Values"] = many_categorical_columns
        else:
            many_categorical_columns = pipeline["Cat Columns with Many Unique Values"]

        X = X.drop(many_categorical_columns, axis=1)

    # Categorical Encoding
    if(fe_sets["fe_cat_encoding"]):
        categorical_columns = list(X.select_dtypes(include=['object', 'bool']).columns.values)
        if(fe_sets["fe_cat_type"]=="One-hot"):
            if(split=="Train"):
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='infrequent_if_exist')
                caten_X_features = encoder.fit_transform(X[categorical_columns])
                pipeline["Cat Encoder"] = encoder
            else:
                encoder = pipeline["Cat Encoder"]
                caten_X_features = encoder.transform(X[categorical_columns])

            cat_feature_names = encoder.get_feature_names_out(categorical_columns)
            caten_X_features = pd.DataFrame(caten_X_features, columns=cat_feature_names)

            X = pd.concat([X, caten_X_features], axis=1)
            X = X.drop(categorical_columns, axis=1)
        elif(fe_sets["fe_cat_type"]=="Categorical"):
            if(split=="Train"):
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
                encoder.set_output(transform="pandas")
                encoder = encoder.fit(X[categorical_columns])
                pipeline["Cat Encoder"] = encoder
            else:
                encoder = pipeline["Cat Encoder"]
            cat_cols = encoder.transform(X[categorical_columns])

            for i, name in enumerate(categorical_columns):
                cat_cols[name] = cat_cols[name].fillna(-1)
                cat_cols[name] = pd.Categorical.from_codes(
                    codes=cat_cols[name].astype(np.int32), categories=encoder.categories_[i]
                )
            X[categorical_columns] = cat_cols

    X = X.rename(columns=lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))

    # # Resampling
    # if(resampling and split=="Train"):
    #     pass

    # Feature Scaling
    feature_names = X.columns.tolist()
    if(fe_sets["fe_scaling"]):
        numerical_columns = list(X.select_dtypes(include=[np.number]).columns.values)
        if(split=="Train"):
            scaler = StandardScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
            pipeline["Scaler"] = scaler
        else:
            scaler = pipeline["Scaler"]
            X[numerical_columns] = scaler.transform(X[numerical_columns])

    # Return
    if(split=="Train"):
        return X, y, ids, feature_names, pipeline
    elif(split=="Valid"):
        return X, y, ids
    else:
        return X, ids