# EM for missing data imputation on the Framingham Heart Study dataset

**Dataset choice:** Ashish Bhardwaj. (2022). Framingham heart study dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DSV/3493583​​

## **Goal:**
- Achieve data points to train a Coronary Heart Disease in 10 years (CHD) classification model and not remove faulty or null data.​ A way might exist to doing missing value imputation without losing important information. The missing/null values shall be tried to be imputed using the Expectation-Maximization (EM) algorithm. 

## **Implementation**
- Done in Python.​
- Using the libraries SciPy (for the EM algorithm itself) and NumPy (numerical processing)​
- Pandas, Matplotlib for data visualization​
- Using a LogisticRegression module from `sklearn.linear_model`.

## **Planned approach:**

> Assumptions: A mixture of Gaussian Normal distributions shall be fit on the known features, to hopefully map to find unknown features. Additionally, it is assumed that the missingness in the dataset is fully at random. 

The EM algorithm can require either i) the missing data, or ii) the underlying distribution.​
- At a particular data point n, treat the features we know as $x$ (age, BMI, sysBP, diaBP) and the features ​which have 'nan' as $z$ (e.g. totChol and heartRate).​
- Get a joint Normal distribution (Mixture of Gaussians) of $x$ and $z$, from non-null rows of the dataset.​
- Using known corollaries for Gaussian distributions (Bayesian approach), translate it to a conditional posterior form for the $z$ values given $x$, and find mean + variance.​
- Use the mean of the posterior to fill the missing feature at row n.​

Try these steps at a new row, using parameters estimated from row 1, and now the filled full row's data. 

- Compute the expected log-likelihood.​​
- Run the M-step: ​Find new parameters that maximize L $\rightarrow$ Update posterior's parameters (based on new gained understanding).​ ​​
- Then the mean of the updated posterior is now filled in place of 'nan' at row 2. ​

Repeat until all null values across a feature have been filled.

----------------

Upon reaching the last row, we will have:
- taken in information of EM updates from all imputed + observed points thus far​
- optimized the overall posterior accordingly, with the hope of giving better and closer imputations.
