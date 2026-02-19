This project analyzes telecom customer data to understand why customers leave a service and to build a predictive model for churn. I cleaned and prepared the dataset by handling data type issues, encoding categorical variables, and structuring the data for modeling. Through exploratory analysis, I identified that customers with low tenure, month-to-month contracts, and higher monthly charges are more likely to churn.

I built and compared Logistic Regression, Random Forest, and Gradient Boosting models to evaluate predictive performance. Instead of focusing only on accuracy, I prioritized recall for churned customers, since identifying at-risk users is more important in a retention context. After comparing models, I selected Logistic Regression and tuned the decision threshold to 0.30 to improve churn detection.

The final model balances interpretability and business relevance, demonstrating how model evaluation and threshold adjustment can meaningfully improve practical outcomes.
