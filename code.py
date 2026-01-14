# Imports
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

###################### A. for reading the dataset

df = pd.read_csv("framingham.csv")
print(df.columns)
print(len(df[df.glucose.isna()]))
print(len(df[df.totChol.isna()]))
print(len(df[df.glucose.isna() & df.totChol.isna()]))

# was done separately in an R environment

#install.packages("mice")
#dat = read.csv("framingham.csv", header=TRUE)
#dat$education <- NULL
#mice::md.pattern.png
Image("mice::md.pattern.png") # missingness pattern

###################### B. for printing number of rows with and without missing values

# total number of rows of the dataset
print("number of rows", len(df)) # number of rows

# number of rows without missing values
df_non_null = df.dropna()
print("number of rows without missing values", len(df_non_null), '\n----------------')

###################### C. the mice implementation

def mice(columns_of_interest=['glucose', 'totChol'], T=20):
    df0 = df.copy() # initialize the first copy
    df_list = [df0.copy() for _ in range(T)] # these will eventually be replaced
    models = [] # store models at each iteration

    means = np.zeros((T, len(columns_of_interest)))
    sd_s = np.zeros((T, len(columns_of_interest)))

    for i in range(1,T):
        for col in columns_of_interest:
            df1 = df_list[i-1].dropna(subset=[col])
            nulls_tempimputed = df1[df1.columns.difference([col])].fillna(df1.median())

            X = pd.get_dummies(nulls_tempimputed, drop_first=True)
            X = nulls_tempimputed.to_numpy()
            y = df1[col].to_numpy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if len(models) == 0:
                model = RandomForestRegressor(n_jobs=-1, random_state=42)
            else:
                model = models[-1]
            model.fit(X_train, y_train)
            
            # Predict missing values in the current dataframe
            df1 = df_list[i].copy()

            # first: find where nulls of the column are, then drop rows where other columns are null
            # because we need the rest of the columns to be non-null to do the prediction
            null_rows = df1[df1[col].isna()].dropna(subset=df1.columns.difference([col]))

            # find where to do the imputations
            d1 = df1[df1[col].isnull()]
            d2 = d1[d1[df1.columns.difference([col])].notna().all(axis=1)]
            
            # perform the imputation
            df1.loc[d2.index, col] = model.predict(null_rows[null_rows.columns.difference([col])].dropna().to_numpy())
            
            models.append(model)
            df_list[i] = df1.copy()
            means[i, columns_of_interest.index(col)] = df1[col].mean()
            sd_s[i, columns_of_interest.index(col)] = df1[col].std()
        
        #print("Null values before in", col, ":", len(df_list[i-1][df_list[i-1][col].isna()]), " after:", len(df1[df1[col].isna()]))

    return df_list[-1], models[-1], means, sd_s

###################### D. for plotting the means and standard deviations of the features over 40 MICE-iterations

trials = 5
plt.xlabel("Iteration")
plt.ylabel("Mean of glucose in MICE")
for i in range(trials):
    df_imputed, model, means, sd_s = mice(columns_of_interest=['glucose'], T=40)
    plt.plot(range(len(means[1:,0])), means[1:,0])
plt.show()
plt.ylabel("St.Dev. of glucose in MICE")
for i in range(trials):
    df_imputed, model, means, sd_s = mice(columns_of_interest=['glucose'], T=40)
    plt.plot(range(len(means[1:,0])), sd_s[1:,0])
plt.show()
plt.ylabel("Mean of totChol in MICE")
for i in range(trials):
    df_imputed, model, means, sd_s = mice(columns_of_interest=['totChol'], T=40)
    plt.plot(range(len(means[1:,0])), means[1:,0])
plt.show()
plt.ylabel("St.Dev. of totChol in MICE")
for i in range(trials):
    df_imputed, model, means, sd_s = mice(columns_of_interest=['totChol'], T=40)
    plt.plot(range(len(means[1:,0])), sd_s[1:,0])
plt.show()

###################### E. difference in accuracy of logistic regression models
###################### when trained on datasets with different missing value handling methods
###################### (1. removing nulls, 2. mean imputation, 3. implemented MICE method)

old_dataset = df[df.columns.difference(['education'])].dropna()
print("Removing null values -->", len(old_dataset))
X = old_dataset[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']].to_numpy()
y = old_dataset['TenYearCHD'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("train accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))

df_impute_with_mean = df.copy()
df_impute_with_mean[['totChol', 'glucose']] = df[['totChol', 'glucose']].fillna(df.mean())
print("Pandas method of filling null values with mean -->", len(df_impute_with_mean))
set_ = df_impute_with_mean[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD']].dropna().to_numpy()
X = set_[:, :-1]
y = set_[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("train accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))

df_imputed, model, means, sd_s = mice(columns_of_interest=['glucose', 'totChol'], T=40)
df_imputed = df_imputed[df_imputed.columns.difference(['education'])].dropna()
print("Replacing values by MICE -->", len(df_imputed))
X = df_imputed[['male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose']].to_numpy()
y = df_imputed['TenYearCHD'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(solver='newton-cg', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("train accuracy:", model.score(X_train, y_train))
print("test accuracy:", model.score(X_test, y_test))

######################