# Missing Data Imputation on the Framingham Heart Study Dataset

Aditya Khadkikar, Uppsala University.

> Code File (consisting of sections labelled as 'appendices') - [code.py](code.py)

# **Introduction**

The dataset used is the Framingham study dataset, which is posed as a classification problem of predicting whether a patient is likely or not to have​ Coronary Heart Disease (CHD) in 10 years.​ It consists of nominal and continuous feature variables such as Male, currentSmoker, BPMeds, prevalentStroke, prevalentHyp, diabetes (Binary ($\{0,1\}$))​​, Education (Ordinal Int 1-4)​​, Age (Continuous Int)​​, and cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, glucose (Continuous Float). The target variable is labelled as TenYearCHD (Binary)​​.

Datasets can be under-maintained, or there can be missed points or features during data collection.​ Patient tests may be infeasible or incomplete, leading to some points being null.​ Upon analysing the dataset, null values were found in certain crucial features (e.g. 50 null values in the totChol feature, 19 in the BMI feature, and up to 388 in the glucose feature), which could be argued that play, if not significant, a role in effectively predicting the risk of coronary heart disease.

# **Purpose**
There are multiple ways to approach this problem of incomplete data fields; one could replace them with the mean of the points of that feature, or simply eliminate the rows where the null values are situated. However, that would result in fewer data points to use in training, which is not desired in curating data for training.

Through this project, the goal that is to be met is to replace the null data with estimated imputations to train a classification model, and make use of the rows in order to not remove existing data.​ Imputation by Chained Equations is the attempted method in this project to do missing value imputation without losing important information, and use the patterns from available data in rest of the features to fill data.​

Research Questions:

> - Can we estimate missing/null glucose and cholesterol values using Imputation by Chained Equations?

> - How much does the mean and standard deviation of the imputed column vary with imputed values?

> - Can we improve a CHD binary prediction model's test accuracy using the new simulated data points?

# **Statistical Method**

The algorithm works by iteratively, for upto $T$ number of ICE iterations, $T$ copies of the original dataset $Y^{(0)}$ are made. For the first iteration, we go by each column we are interested in imputing null values in. The columns of interest are `glucose` and `totChol`. 

We focus on one column of interest at a time. There are two types of possible rows; ones where `glucose`/`totChol` is observed, and those where they are missing. The missing rows are omitted temporarily. On the observed rows, we fit a regression model, setting e.g. `glucose` as a function of the rest of the variables in the dataset. As notation, one can refer to them as features to `glucose`, and equivalently features to `totChol`. As a temporary filling value, we substitute all indexes in features to `glucose`, where a null value is present, with the column mean. Next, we train a regression model to the target, and get weights $\theta_{\text{totChol}}$ and $\theta_{\text{glucose}}$, which is a vector. 

For the first iteration, we have obtained $\theta^{(1)}_{\text{glucose}}$, $\theta^{(1)}_{\text{totChol}}$, and a filled dataset $Y^{(1)}$, which is not yet 'optimum'. Recalling the rows which were omitted for training for $\theta_{(g)}$, we now replace them with the output from the model, which will be a more accurate imputation than the mean, which was a more naive filled value. Repeat this process for all of the columns in which null values are to be imputed.

In the next imputation iteration, we bring the `glucose` and the `totChol` model trained from the previous iteration. We repeat the same procedure we did, in order of the columns of interest.

- `glucose`: `totChol` is the feature to our temporary predictor for `glucose`, so in this second copy of the dataset, instead of filling the null values with the mean like we did in the very start, now fill with the model for `totChol`. 

- `totChol`: Instead of replacing the `glucose` null values with the mean, now they get temporarily filled with the model predictions for `glucose` from this iteration - 1. 

Now, train a fully new linear regression model, and repeat the process by moving the obtained weights to the next onward iteration. Now, we have obtained updated $\theta^{(2)}_{\text{glucose}}$, $\theta^{(2)}_{\text{totChol}}$, and a slightly better filled dataset $Y^{(2)}$, which is not yet 'optimum'. Do for $T$ iterations, with the final imputed dataset $Y^{(T)}$ and models (represented as weights) $\Theta^{(T)}$. Hopefully for doing this a sufficient amount of times, this will converge closer to the ideal values having taken the patterns from all of the other variables into account. The implementation of the method `mice()` is found in Appendix C.

# **Implementation**

As part of the fitting, it was chosen to drop the 'education' feature. As a step above a simple Linear Regression, a RandomForestRegressor from `sklearn` was chosen for the model type trained at each imputation step, as it can deal with non-linearity in a better manner. For the entire code, please see the appendix at the end of the report. 

# **Dataset Analysis**

The total number of rows in this dataset are 4,240, with 3,658 rows if any row with missing value in any column were to be omitted. 582 training examples are taken out of the dataset, as a result. In the image below, the missingness in each column type is shown, where rows can have up to 3 features missing in the same training example. 388 missing observations are present for the `glucose` column, 50 for the `totChol` column, and 40 data points where both fields are NA (see Appendix A).


```python
from IPython.display import Image, display
display(Image('mice::md.pattern.png'))
```


    
![png](Individual%20Project%201MS049%20Computer%20Intensive%20Statistics_files/Individual%20Project%201MS049%20Computer%20Intensive%20Statistics_3_0.png)
    


# **Comparison with other null value handling methods**

Upon running the ICE algorithm, the goal was to now inspect whether the imputed values follow a healthy convergence (van Buuren, p.37), by conducting 5 simulations of the ICE algorithm in filling the null values based on derived relationships from the rest of the variables. The means and standard deviations of both the features of interest converge substantially around a small region in amplitudes of not greater than 0.1. For the graph generation code, see Appendix D.


```python
from IPython.display import Image, display
display(Image('plots.png'))
```


    
![png](Individual%20Project%201MS049%20Computer%20Intensive%20Statistics_files/Individual%20Project%201MS049%20Computer%20Intensive%20Statistics_6_0.png)
    


This can be quite well in doing null-value imputations for datasets where several such entries exist, and with each ICE iteration in $1:T$, the model improves with the previous filled dataset using as its training data in the subsequent iteration, and the model moving onward in the next iteration. The graphs also suggest that the stabilization point can be reached, with not requiring more than 10 to 15 iterations. Additionally, the relationships with the remaining features are taken into account in providing the imputation. 

However, a stabilization point is not always guaranteed, as there could be over-dependency between variables. The method can pose problems if data is ordered (van Buuren, p.6), and this was addressed by using sklearn's train-test-split method of dataset preparation at each iteration, and shuffles the training and testing splits respectively. The method can also take longer computation time the more complex the choice of the imputation model is. Additionally, the choice of the model that is fit to find the feature of interest, can give very different outcomes, and majorly depends on the nature of the data, and the missingness. A simplification assumption was made in this study that the data was missing at random, but data in practice is not necessarily so. 

Lastly, for conducting in practice in the most effective way, it would be confirming the validity of the newly points would be with the domain specialists, such as, in this case, healthcare professionals. Otherwise, there can be a false replacement procedure in place, and can predict an overly strange value for e.g. `heartRate` (a value too high/low possible for a patient).

# **Conclusion**

From the initially presented research questions, it was shown that it is possible to do null value imputations in the `totChol` and `glucose` features of the Framingham Heart Study dataset using Imputation by Chained Equations, in order to get more useful values with which training can be conducted. With the null-value handling, an additional of 340 training examples were achieved, compared to if null-values were to be dropped entirely. The graphs show the means and standard deviations varying with up to 40 imputation iterations done, for 5 simulations, reaching a somewhat stable region. 

As a final step, a logistic regression model from `sklearn` was trained on 3 methods of null-value handling (see Appendix E); 1) removing null values altogether (86.1% train and 83.7% test accuracy), 2) filling them with simply the column mean (85.8% train and 84.9% test accuracy), and 3) the implemented imputation algorithm (85.85% train and 85.04% test accuracy). It was visible that the algorithm helped in better test accuracy, though impacted training accuracy slightly, and outperformed both method 1) and 2). 

# **References**

Dataset: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset/data

van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software, 45(3), 1–67. https://doi.org/10.18637/jss.v045.i03

Oakley, Jeremy and Poulston, Alison (2021). MAS61006 Bayesian Statistics and Computational Methods. Semester 2: Computational Methods, University of Sheffield. "5. Multiple Imputation for missing data". https://oakleyj.github.io/MAS61006Sem2/multiple-imputation-for-missing-data.html