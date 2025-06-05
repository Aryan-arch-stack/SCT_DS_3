# SCT_DS_3
📌 Project Objective
In today's competitive and highly digitized financial sector, understanding customer behavior is no longer a luxury—it's a necessity. Banks and financial institutions worldwide invest substantial resources into marketing campaigns aimed at promoting specific products or services, such as term deposits. However, the traditional “one-size-fits-all” marketing approach often results in inefficient use of resources, low conversion rates, and reduced return on investment. To address these challenges, data-driven strategies are being increasingly adopted, where historical customer interaction data is analyzed to generate insights and guide future decisions.

This project is built around a real-world dataset from a Portuguese bank that conducted a direct marketing campaign through phone calls, targeting clients for a term deposit product. The goal of this campaign was to convince customers to subscribe to a long-term deposit, which is beneficial both for the customer and the bank in terms of returns and financial stability, respectively. However, like most marketing campaigns, not every client responded positively, and many declined the offer.

The primary objective of this project is to develop a predictive machine learning model that can analyze patterns in historical client data and forecast whether a customer is likely to subscribe to a term deposit in response to a future campaign. By identifying these likely subscribers in advance, the bank can refine its marketing strategy, target the right group of customers, reduce unnecessary outreach, and improve campaign success rates. This approach not only saves time and cost but also enhances customer experience by reducing irrelevant or repetitive communication.

We chose a **Decision Tree Classifier** as the initial model for its interpretability, ease of use, and suitability for categorical as well as numerical data. The decision tree structure allows stakeholders, including those without technical expertise, to visualize and understand how the model is making decisions based on customer attributes like age, job type, marital status, education level, and more.

This project is not just an academic exercise but a simulation of a real-world banking problem where predictive analytics can significantly influence strategic decisions. Through careful preprocessing, feature selection, model training, and evaluation, we aim to demonstrate how a relatively simple machine learning algorithm can add meaningful value to a business use case. Additionally, by evaluating the model's performance—especially in the face of challenges like class imbalance—we hope to highlight practical limitations and propose improvements that reflect industry best practices.


📊 Dataset Overview
Dataset Name: Bank Marketing Dataset
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
File Used: `bank-full.csv`
Total Records: 45,211
Features: 17 input features + 1 target variable
Target Variable: `y` — whether the client subscribed to a term deposit (`yes` or `no`)

The Bank Marketing Dataset is a widely studied real-world dataset, originating from a Portuguese banking institution’s direct marketing campaigns. These campaigns were conducted via phone calls and aimed to encourage existing clients to subscribe to a long-term term deposit product. The dataset captures rich information about client demographics, socio-economic indicators, and details about previous marketing interactions. With over 45,000 records and a broad range of features, it provides a realistic and valuable context for building machine learning models focused on binary classification.

🔍 Data Characteristics
The dataset consists of 17 input features, both categorical and numerical in nature. These can be grouped broadly into:
1. Client Demographics:
  `age`: Numerical. Represents the client’s age.
  `job`: Categorical. Type of job (e.g., technician, blue-collar, entrepreneur).
  `marital`: Categorical. Marital status (married, single, divorced).
  `education`: Categorical. Education level (primary, secondary, tertiary, unknown).
  `default`: Categorical. Has credit in default?
  `balance`: Numerical. Average yearly balance in euros.
  `housing`: Categorical. Has a housing loan?
  `loan`: Categorical. Has a personal loan?

2. Campaign Contact Data:
  `contact`: Categorical. Type of communication contact (cellular, telephone, unknown).
  `day`: Numerical. Last contact day of the month.
  `month`: Categorical. Last contact month of year (e.g., may, jun, jul).
  `duration`: Numerical. Last contact duration, in seconds.

3. Campaign Performance Data:
  `campaign`: Numerical. Number of contacts performed during this campaign.
  `pdays`: Numerical. Number of days since the client was last contacted (999 = never).
  `previous`: Numerical. Number of contacts performed before this campaign.
  `poutcome`: Categorical. Outcome of the previous marketing campaign (success, failure, non-existent).

4. Target Variable:
 `y`: Categorical (binary). Indicates whether the client subscribed to a term deposit (`yes` or `no`).

🧠 Use Case Relevance
This dataset is particularly useful for machine learning projects involving:
  -Predictive modeling (classification),
  -Marketing optimization, and
  -Customer segmentation.
The high imbalance in the target variable (majority class being "no") introduces real-world complexity in modeling. This necessitates not only solid algorithmic understanding but also proper handling of class imbalance, feature engineering, and performance evaluation beyond accuracy.
Because of its structured nature and business relevance, the dataset is a popular choice for building interpretable models like Decision Trees, as well as for comparing performance with more complex models like Random Forests or Gradient Boosted Trees.


🔧 Project Workflow & Implementation
Creating a robust machine learning pipeline to predict customer behavior from raw banking data involves several critical steps — from ingestion and preprocessing to model selection, training, evaluation, and insight extraction. This section outlines how we systematically tackled each phase of the project, along with the challenges encountered and how we addressed them to derive meaningful outcomes.

1. Data Upload & Extraction
The initial phase of the project began with acquiring and preparing the data. The dataset was provided in `.zip` format (`bank.zip`), which we uploaded and extracted in a Jupyter/Colab environment. The key file used in this study was `bank-full.csv`, which contains over 45,000 records and a mix of 17 input features along with a binary target variable indicating whether a client subscribed to a term deposit.
We used Pandas to read the data:
```python
import pandas as pd
df = pd.read_csv('bank-full.csv', sep=';')
```
Upon loading, the first inspection revealed that the data included a rich combination of categorical and numerical attributes such as `job`, `marital`, `education`, `age`, `balance`, and more.
This step also involved ensuring the dataset was clean — no missing values were found in the key features, but we made mental notes to revisit imbalanced class distribution in the target column later during model evaluation.

2. Data Preprocessing
✔️ Challenge: Categorical Variables Not Directly Usable
Scikit-learn models require all features to be numerical. However, the dataset included several categorical fields such as:
`job`, `marital`, `education`, `housing`, `loan`, `contact`, `month`, `poutcome`, etc.
If passed directly to the model, these would cause errors or misleading inferences.

🛠️ How We Tackled It:
We used **one-hot encoding** via `pandas.get_dummies()` to transform these categorical features into binary flags. For example, the `job` field which contains values like 'admin.', 'technician', 'services', etc., was converted into multiple columns like `job_admin.`, `job_technician`, and so on.
To reduce redundancy and avoid multicollinearity, we dropped the first level of each categorical field:
```python
df_encoded = pd.get_dummies(df, drop_first=True)
```

🧠 Feature Selection & Target Definition
We defined the target variable `y` as a binary field:
* `0` if the client did not subscribe
* `1` if the client did subscribe
The features (`X`) were selected from the encoded dataset by dropping the column `y_yes`, which was extracted as the binary target (`y = df_encoded['y_yes']`).
We then split the dataset into training and testing subsets using an 80-20 ratio with a fixed `random_state` to ensure results were reproducible across experiments.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

3. Model Training: Decision Tree Classifier
✔️ Challenge: Need for Interpretability
In a banking or financial setting, models aren’t just evaluated based on accuracy — they must be nterpretable. Business stakeholders should understand why a model makes a particular decision, especially in cases involving customer decisions or risk profiles.
🛠️ Solution: Decision Tree
We chose DecisionTreeClassifier from scikit-learn due to its:
Simplicity
Visual interpretability (via tree diagrams)
Ability to handle both numerical and categorical (encoded) features
To prevent overfitting and improve readability, we constrained the depth of the tree:
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4, random_state=42)
clf.fit(X_train, y_train)
```
This limited depth allowed the tree to capture major decision ruleswhile remaining visually interpretable when plotted.

4. Model Evaluation
Once trained, we used the test set to evaluate the model's performance. The key metrics used were:
  Accuracy
  Precision
  Recall
  F1-Score
  Confusion Matrix

🧪 Results:
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

📊 Accuracy: 87.93%
At first glance, this seems impressive. However, digging deeper into the confusion matrix and recall metrics, we observed a significant challenge:
🔎 Confusion Matrix:
```
[[7949    3]
 [1088    3]]
```
  The model correctly predicted 7949 out of 7952 "no" cases.
  However, it only correctly predicted 3 out of 1091 "yes" cases.

📉 Classification Report:
  -Precision (No): 87.9%
  -Recall (No): 99.9%
  -Precision (Yes): 50.0%
  -Recall (Yes): 0.3% 🔴
This clearly shows a strong class imbalance: The model overwhelmingly favors predicting "no" because the dataset is heavily skewed in that direction. The model almost completely fails to identify true positives — which, in this context, means it can’t effectively predict who will subscribe to the term deposit.


📉 Challenge Faced: Severe Class Imbalance
One of the most significant challenges we encountered during this project was the severe class imbalance present in the target variable `y`, which indicates whether a customer subscribed to a term deposit. Upon inspecting the distribution of values in this column, we found that approximately 88% of the clients responded with "no", while only 12% responded with "yes".

Such an imbalance can dramatically affect the performance and fairness of machine learning models. Standard classification algorithms — especially ones like Decision Trees — tend to become biased toward the majority class. As a result, the model may achieve high accuracy simply by always predicting the majority outcome ("no"), but this comes at the cost of failing to correctly identify minority class instances, which in this case are the most business-critical: clients who actually subscribed.

⚠️ Why Is This a Problem?
In the context of a marketing campaign, the ultimate goal is **not just to predict correctly in general**, but to **accurately identify potential customers** who are likely to convert (i.e., say "yes"). If the model only predicts “no” with high accuracy but misses the rare “yes” predictions, it becomes nearly useless from a marketing strategy perspective.

🛠️ What We Did:
1. Went Beyond Accuracy
While the initial model showed a promising overall accuracy of 87.93%, we immediately recognized this as potentially misleading due to the imbalance. A deeper look into the confusion matrix and classification report confirmed our concern:
```
Confusion Matrix:
[[7949    3]
 [1088    3]]
```
Here’s what this means:
  True Negatives (Correctly predicted “no”): 7949
  False Positives (Incorrectly predicted “yes”): 3
  False Negatives (Missed “yes”): 1088
  True Positives (Correctly predicted “yes”): 3
Clearly, the model was heavily biased toward predicting "no" and almost never caught the true "yes" cases.
The classification report further illustrated this imbalance:
```
Precision (Yes): 50.0%
Recall (Yes): 0.3%
F1-score (Yes): 0.6%
```
While the model was technically precise (it was right about “yes” when it rarely predicted it), it was **almost completely failing to recall** any actual “yes” cases. This low recall made the model unsuitable for its intended purpose.

2. Plotted the Decision Tree
To better understand the model’s behavior and what decisions it was making, we used `plot_tree()` from `sklearn.tree`:
```python
from sklearn.tree import plot_tree
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["No", "Yes"])
```
The visualization gave us insight into the split criteria at each node — such as balance thresholds, contact type, duration, and previous outcome — but it also made it clear that despite a well-structured decision path, the tree was skewed by the overwhelming presence of the "no" class during training.

✅ Outcome and Takeaway
This imbalance highlighted a key learning: high accuracy can be deceptive in imbalanced classification tasks. By digging deeper into recall, precision, and F1-scores, we were able to uncover the model’s real-world limitations. This informed our next steps: to incorporate strategies like class weighting, resampling (SMOTE/undersampling), or trying ensemble methods (Random Forest, XGBoost) that are more resilient to imbalance.
Addressing this imbalance is crucial for building a truly effective predictive model — especially in applications like banking, healthcare, or fraud detection, where the minority class often carries the highest value.


🧠 Key Learnings and Insights
As we progressed through the project, we discovered that machine learning isn’t just about building models and checking accuracy—it’s about understanding what the model actually learns, and whether those learnings align with our goals. This project served as a powerful reminder that high accuracy does not necessarily mean high performance, especially when working with imbalanced datasets like ours.

At first glance, the decision tree model seemed to perform well, boasting an accuracy of 87.93%. However, upon closer inspection, we realized that this number was misleading. Why? Because the dataset was heavily skewed toward one class—**the majority of customers said “no” to the term deposit. This imbalance led to a model that was very good at identifying “no” responses, but almost completely failed to catch the rare but critical “yes” cases.

This issue came into sharp focus when we examined the confusion matrix and classification report. Despite having high overall accuracy, the model’s recall for the positive class (“yes”) was almost 0.3%, meaning that out of more than 1,000 actual “yes” cases, the model identified only 3 correctly. The rest were completely misclassified as “no”. This failure was significant—it underscored how easy it is for a model to appear successful while missing the very insights we care most about.

To better understand how our decision tree was functioning, we used `plot_tree()` to visualize the decision-making process. This step proved incredibly useful, allowing us to see which features the model prioritized when making predictions. We noticed that features such as duration, contact type, previous campaign outcome, and balance played significant roles in determining the splits. This visualization not only added transparency to our model but also guided us in thinking about potential improvements.

Through this, we learned that model interpretability is just as important as performance metrics. A decision tree is easy to understand and explain to stakeholders, but if it’s misled by imbalanced data, its insights can be flawed. Therefore, we began exploring ways to address this imbalance.

🛠️ Strategies Considered for Improvement:
1. Setting `class_weight='balanced'` in the model
   This tells the algorithm to give more importance to the minority class during training, helping it pay closer attention to “yes” responses.

2. Resampling Techniques
   We considered:
     Oversampling the minority class using techniques like **SMOTE (Synthetic Minority Over-sampling Technique)**, which generates synthetic examples based on existing minority samples.
     Undersampling the majority class to reduce its dominance in the training data.
   
4. Trying Ensemble Methods
   Decision Trees alone may be too simple to handle complex patterns in imbalanced data. We plan to experiment with:
     Random Forest: Aggregates multiple decision trees for improved generalization.
     XGBoost: Known for handling class imbalance effectively and delivering strong predictive power.


Conclusion
This project showcased the value of predictive modeling in enhancing marketing strategies within the banking sector. Using historical data from a Portuguese bank’s direct marketing campaign, we set out to predict whether a client would subscribe to a term deposit. The initial model — a Decision Tree Classifier — achieved an accuracy of nearly 88%. However, further evaluation uncovered a significant flaw: it failed to effectively predict positive cases, i.e., actual subscribers.

This issue stemmed from a severe class imbalance in the dataset — around 88% of clients did not subscribe, and only 12% did. As a result, the model leaned heavily toward the majority class, leading to a high number of false negatives. While the model appeared accurate overall, it lacked practical utility in identifying the very customers the bank aimed to target.
This experience highlighted a key lesson: accuracy alone is not a sufficient metric, especially when dealing with imbalanced datasets. By analyzing precision, recall, and the F1-score, we gained a clearer picture of performance and identified areas for improvement. Visualizing the decision tree also helped us understand how predictions were made and which features influenced outcomes the most.

Recognizing this limitation has set the stage for improvement. In future iterations, we plan to apply class weighting, oversampling (e.g., SMOTE), and explore ensemble models like Random Forest and XGBoost to better handle imbalance and improve predictions on the minority class.
Ultimately, this project underlines the iterative nature of machine learning. Initial models act as a baseline, exposing challenges and guiding refinement. With each step, we move closer to a model that not only performs well statistically but also delivers real-world value in targeted marketing and customer engagement.
