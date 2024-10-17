from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Define the path to save uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create a static directory for storing images
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Function to generate graphs
def generate_graphs(data, label_encoders, cnf_matrix, y_test, prediction, logreg, X_test):
    # Count of Attrition
    plt.figure(figsize=(15, 5))
    sns.countplot(y='Attrition', data=data)
    plt.title('Count of Attrition')
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_count.png'))
    plt.close()

    # Attrition by Department
    plt.figure(figsize=(12, 5))
    departments = label_encoders['Department'].inverse_transform(data['Department'])
    sns.countplot(x=departments, hue='Attrition', data=data, palette='hot')
    plt.title("Attrition w.r.t Department")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_department.png'))
    plt.close()

    # Attrition by Education Field
    plt.figure(figsize=(12, 5))
    edu_field = label_encoders['EducationField'].inverse_transform(data['EducationField'])
    sns.countplot(x=edu_field, hue='Attrition', data=data, palette='hot')
    plt.title("Attrition w.r.t EducationField")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_education_field.png'))
    plt.close()

    # Job Role w.r.t Attrition
    plt.figure(figsize=(12, 5))
    job_roles = label_encoders['JobRole'].inverse_transform(data['JobRole'])
    sns.countplot(x=job_roles, hue='Attrition', data=data, palette='hot')
    plt.title("JobRole w.r.t Attrition")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_job_role.png'))
    plt.close()

    # Gender w.r.t Attrition
    plt.figure(figsize=(12, 5))
    sns.countplot(x='Gender', hue='Attrition', data=data, palette='hot')
    plt.title("Gender w.r.t Attrition")
    plt.xticks(ticks=[0, 1], labels=['Male', 'Female'])
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_gender.png'))
    plt.close()

    # Age distribution
    plt.figure(figsize=(12, 5))
    sns.histplot(data['Age'], kde=True)
    plt.title("Age Distribution")
    plt.savefig(os.path.join(STATIC_FOLDER, 'age_distribution.png'))
    plt.close()

    # Education w.r.t Attrition
    edu_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}
    plt.figure(figsize=(12, 5))
    sns.countplot(x=data['Education'].map(edu_map), hue='Attrition', data=data, palette='hot')
    plt.title("Education W.R.T Attrition")
    plt.savefig(os.path.join(STATIC_FOLDER, 'attrition_education.png'))
    plt.close()

    # Confusion Matrix and ROC Curve
    fig = plt.figure(figsize=(15, 6))
    
    # Confusion Matrix
    ax1 = fig.add_subplot(1, 2, 1)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='d', ax=ax1)
    bottom, top = ax1.get_ylim()
    ax1.set_ylim(bottom + 0.5, top - 0.5)
    plt.xlabel('Predicted')
    plt.ylabel('Expected')

    # ROC Curve
    ax2 = fig.add_subplot(1, 2, 2)
    y_pred_proba = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)
    ax2.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.legend(loc=4)
    
    # Save the plot
    plt.savefig(os.path.join(STATIC_FOLDER, 'confusion_matrix_roc.png'))
    plt.close()

# Route for home page and file upload
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and file.filename.endswith('.csv'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Read the CSV file
            data = pd.read_csv(file_path)

            # Data preprocessing
            data['Attrition'] = data['Attrition'].replace({'No': 0, 'Yes': 1})
            data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})
            data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

            # Encode categorical variables
            encoding_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
            label_encoders = {}
            for column in encoding_cols:
                label_encoders[column] = LabelEncoder()
                data[column] = label_encoders[column].fit_transform(data[column])

            # Resampling to handle class imbalance
            X = data.drop(['Attrition', 'Over18'], axis=1)
            y = data['Attrition'].values
            rus = RandomOverSampler(random_state=42)
            X_over, y_over = rus.fit_resample(X, y)

            # Split the data into training and testing set
            X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)

            # Logistic Regression
            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            prediction = logreg.predict(X_test)

            # Confusion matrix
            cnf_matrix = confusion_matrix(y_test, prediction)

            # Generate graphs (including confusion matrix and ROC curve)
            generate_graphs(data, label_encoders, cnf_matrix, y_test, prediction, logreg, X_test)

            # Render the table and display the graphs
            return render_template('table.html', column_names=data.columns.values,
                                   row_data=list(data.values.tolist()), zip=zip)

    return render_template('index.html')

# Route to display the graphs
@app.route("/graphs")
def graphs():
    return render_template('graphs.html')

if __name__ == "__main__":
    app.run(debug=True)
