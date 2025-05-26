import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.svm import SVC
import joblib
import io
import base64

st.set_page_config(
    page_title="Data Insights & ML Prediction App",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #4F8BF9;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #888888;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'numerical_columns' not in st.session_state:
    st.session_state.numerical_columns = []

st.markdown("<h1 class='main-header'>Data Insights & ML Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text'>Upload your dataset, visualize it, train a machine learning model, and make predictions!</p>", unsafe_allow_html=True)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Upload Dataset", "Data Exploration", "Data Visualization", "Model Training", "Make Predictions"])

def get_csv_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def get_model_download_link(model, filename="model.joblib"):
    output = io.BytesIO()
    joblib.dump(model, output)
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:file/joblib;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

def preprocess_data(df, target_column, feature_columns, is_training=True):
    X = df[feature_columns].copy()
    
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
    st.session_state.categorical_columns = categorical_columns
    
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    st.session_state.numerical_columns = numerical_columns
    
    for col in categorical_columns:
        if is_training:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            st.session_state.label_encoders[col] = le
        else:
            le = st.session_state.label_encoders.get(col)
            if le:
                # Handle unseen categories
                X[col] = X[col].map(lambda x: 'unknown' if x not in le.classes_ else x)
                # Replace 'unknown' with the most frequent class
                if 'unknown' in X[col].values:
                    most_frequent = X[col].value_counts().index[0]
                    X[col] = X[col].replace('unknown', most_frequent)
                X[col] = le.transform(X[col])
    
    # Scale numerical features
    if len(numerical_columns) > 0:
        if is_training:
            scaler = StandardScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
            st.session_state.scaler = scaler
        else:
            scaler = st.session_state.scaler
            if scaler:
                X[numerical_columns] = scaler.transform(X[numerical_columns])
    
    if target_column:
        y = df[target_column].copy()
        if df[target_column].dtype == 'object' or df[target_column].dtype.name == 'category':
            if is_training:
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.session_state.label_encoders[target_column] = le
            else:
                le = st.session_state.label_encoders.get(target_column)
                if le:
                    y = le.transform(y)
        return X, y
    
    return X

# Upload Dataset
if app_mode == "Upload Dataset":
    st.markdown("<h2 class='section-header'>Upload Your Dataset</h2>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"Dataset successfully loaded with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            st.markdown("<h3>Dataset Preview</h3>", unsafe_allow_html=True)
            st.dataframe(data.head())
            
            st.markdown("<h3>Dataset Information</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<h4>Shape: {data.shape[0]} rows × {data.shape[1]} columns</h4>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h4>Column Types:</h4>", unsafe_allow_html=True)
                for dtype in data.dtypes.value_counts().items():
                    st.markdown(f"- {dtype[0]}: {dtype[1]} columns", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h4>Missing Values:</h4>", unsafe_allow_html=True)
                missing_data = data.isnull().sum()
                missing_columns = missing_data[missing_data > 0]
                
                if len(missing_columns) > 0:
                    for col, count in missing_columns.items():
                        st.markdown(f"- {col}: {count} missing values ({(count/len(data))*100:.2f}%)", unsafe_allow_html=True)
                else:
                    st.markdown("No missing values found!", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<h3>Summary Statistics</h3>", unsafe_allow_html=True)
            st.dataframe(data.describe().T)
            
            # Option to download cleaned data
            if st.checkbox("Clean dataset (remove rows with missing values)"):
                cleaned_data = data.dropna()
                st.session_state.data = cleaned_data
                st.success(f"Dataset cleaned. Removed {len(data) - len(cleaned_data)} rows with missing values.")
                st.dataframe(cleaned_data.head())
                st.markdown(get_csv_download_link(cleaned_data, "cleaned_data.csv"), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV file to get started.")

# Data Exploration
elif app_mode == "Data Exploration":
    st.markdown("<h2 class='section-header'>Data Exploration</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Column selection
        selected_column = st.selectbox("Select a column to explore:", data.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Column Statistics</h3>", unsafe_allow_html=True)
            
            if data[selected_column].dtype == 'object' or data[selected_column].dtype.name == 'category':
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<h4>Data Type: {data[selected_column].dtype}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Unique Values: {data[selected_column].nunique()}</h4>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<h4>Value Counts:</h4>", unsafe_allow_html=True)
                st.dataframe(data[selected_column].value_counts().reset_index().rename(columns={"index": selected_column, selected_column: "Count"}))
                
                # Visualization for categorical data
                fig = px.pie(data, names=selected_column, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown(f"<h4>Data Type: {data[selected_column].dtype}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Min: {data[selected_column].min()}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Max: {data[selected_column].max()}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Mean: {data[selected_column].mean():.2f}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Median: {data[selected_column].median()}</h4>", unsafe_allow_html=True)
                st.markdown(f"<h4>Standard Deviation: {data[selected_column].std():.2f}</h4>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Visualization for numerical data
                fig = px.histogram(data, x=selected_column, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h3>Missing Values</h3>", unsafe_allow_html=True)
            missing_count = data[selected_column].isnull().sum()
            missing_percentage = (missing_count / len(data)) * 100
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"<h4>Missing Values: {missing_count} ({missing_percentage:.2f}%)</h4>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Correlation with other numerical columns
            if data[selected_column].dtype != 'object' and data[selected_column].dtype.name != 'category':
                st.markdown("<h3>Correlation with Other Numerical Columns</h3>", unsafe_allow_html=True)
                numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                if len(numerical_cols) > 1:
                    correlations = data[numerical_cols].corr()[selected_column].sort_values(ascending=False)
                    correlations = correlations.drop(selected_column)
                    
                    fig = px.bar(
                        x=correlations.values,
                        y=correlations.index,
                        orientation='h',
                        title=f"Correlation with {selected_column}",
                        labels={'x': 'Correlation Coefficient', 'y': 'Features'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No other numerical columns to calculate correlation.")
            
            # Sample data
            st.markdown("<h3>Sample Data</h3>", unsafe_allow_html=True)
            st.dataframe(data[[selected_column]].head(10))
    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# Data Visualization
elif app_mode == "Data Visualization":
    st.markdown("<h2 class='section-header'>Data Visualization</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Visualization type selection
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Histogram", "Box Plot", "Heatmap", "Area Chart"]
        )
        
        if viz_type == "Heatmap":
            st.markdown("<h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numerical_cols) > 1:
                selected_cols = st.multiselect("Select columns for heatmap:", numerical_cols, default=numerical_cols[:min(10, len(numerical_cols))])
                
                if selected_cols:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    correlation_matrix = data[selected_cols].corr()
                    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                    plt.title("Correlation Heatmap")
                    st.pyplot(fig)
                else:
                    st.warning("Please select at least one column.")
            else:
                st.warning("Not enough numerical columns for a heatmap. Need at least 2 numerical columns.")
        
        elif viz_type == "Scatter Plot":
            st.markdown("<h3>Scatter Plot</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if len(numerical_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("Select X-axis:", numerical_cols)
                
                with col2:
                    y_col = st.selectbox("Select Y-axis:", [col for col in numerical_cols if col != x_col])
                
                with col3:
                    color_col = st.selectbox("Color by (optional):", ["None"] + data.columns.tolist())
                
                if color_col == "None":
                    fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                else:
                    fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}, colored by {color_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numerical columns for a scatter plot.")
        
        elif viz_type == "Line Chart":
            st.markdown("<h3>Line Chart</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X-axis:", data.columns.tolist())
                
                with col2:
                    y_cols = st.multiselect("Select Y-axis (multiple):", numerical_cols)
                
                if y_cols:
                    if data[x_col].dtype == 'object' or data[x_col].dtype.name == 'category':
                        st.warning(f"X-axis column '{x_col}' is categorical. Consider using a Bar Chart instead.")
                    
                    fig = px.line(data, x=x_col, y=y_cols, title=f"Line Chart: {', '.join(y_cols)} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one Y-axis column.")
            else:
                st.warning("No numerical columns available for Y-axis.")
        
        elif viz_type == "Bar Chart":
            st.markdown("<h3>Bar Chart</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("Select X-axis (categories):", data.columns.tolist())
            
            with col2:
                y_col = st.selectbox("Select Y-axis (values):", ["Count"] + data.select_dtypes(include=['int64', 'float64']).columns.tolist())
            
            if y_col == "Count":
                # Create count-based bar chart
                value_counts = data[x_col].value_counts().reset_index()
                value_counts.columns = [x_col, 'Count']
                
                fig = px.bar(value_counts, x=x_col, y='Count', title=f"Count of {x_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create value-based bar chart
                fig = px.bar(data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            st.markdown("<h3>Pie Chart</h3>", unsafe_allow_html=True)
            
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                selected_col = st.selectbox("Select column for pie chart:", categorical_cols)
                
                # Limit to top categories if there are too many
                value_counts = data[selected_col].value_counts()
                if len(value_counts) > 10:
                    st.warning(f"Column has {len(value_counts)} unique values. Showing top 10 for clarity.")
                    value_counts = value_counts.head(10)
                    other_count = data[selected_col].value_counts().iloc[10:].sum()
                    value_counts['Other'] = other_count
                
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical columns available for pie chart.")
        
        elif viz_type == "Histogram":
            st.markdown("<h3>Histogram</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    selected_col = st.selectbox("Select column for histogram:", numerical_cols)
                
                with col2:
                    bins = st.slider("Number of bins:", min_value=5, max_value=100, value=20)
                
                fig = px.histogram(data, x=selected_col, nbins=bins, title=f"Histogram of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns available for histogram.")
        
        elif viz_type == "Box Plot":
            st.markdown("<h3>Box Plot</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    y_col = st.selectbox("Select column for box plot:", numerical_cols)
                
                with col2:
                    x_col = st.selectbox("Group by (optional):", ["None"] + data.columns.tolist())
                
                if x_col == "None":
                    fig = px.box(data, y=y_col, title=f"Box Plot of {y_col}")
                else:
                    fig = px.box(data, x=x_col, y=y_col, title=f"Box Plot of {y_col} grouped by {x_col}")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numerical columns available for box plot.")
        
        elif viz_type == "Area Chart":
            st.markdown("<h3>Area Chart</h3>", unsafe_allow_html=True)
            
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Select X-axis for area chart:", data.columns.tolist())
                
                with col2:
                    y_cols = st.multiselect("Select Y-axis columns (multiple):", numerical_cols)
                
                if y_cols:
                    fig = px.area(data, x=x_col, y=y_cols, title=f"Area Chart: {', '.join(y_cols)} vs {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least one Y-axis column.")
            else:
                st.warning("No numerical columns available for area chart.")
    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# Model Training
elif app_mode == "Model Training":
    st.markdown("<h2 class='section-header'>Model Training</h2>", unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Model configuration
        st.markdown("<h3>Model Configuration</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target selection
            target_column = st.selectbox("Select Target Column:", data.columns)
            st.session_state.target_column = target_column
            
            # Task type detection
            if data[target_column].dtype == 'object' or data[target_column].dtype.name == 'category':
                task_type = "Classification"
                st.info(f"Detected task type: Classification (Target has {data[target_column].nunique()} unique values)")
            else:
                unique_values = data[target_column].nunique()
                if unique_values <= 10:
                    task_type = st.radio("Select Task Type:", ["Classification", "Regression"], index=0)
                    if task_type == "Classification":
                        st.info(f"Target has {unique_values} unique values, suitable for classification.")
                    else:
                        st.info("You selected regression despite low number of unique values.")
                else:
                    task_type = "Regression"
                    st.info(f"Detected task type: Regression (Target has {unique_values} unique values)")
            
            st.session_state.model_type = task_type
        
        with col2:
            # Feature selection
            available_features = [col for col in data.columns if col != target_column]
            feature_columns = st.multiselect("Select Feature Columns:", available_features, default=available_features)
            st.session_state.feature_columns = feature_columns
            
            # Test size selection
            test_size = st.slider("Test Size (%):", 10, 50, 20) / 100
        
        # Model selection
        st.markdown("<h3>Model Selection</h3>", unsafe_allow_html=True)
        
        if task_type == "Classification":
            model_name = st.selectbox(
                "Select Classification Model:",
                ["Logistic Regression", "Random Forest", "Support Vector Machine"]
            )
        else:  # Regression
            model_name = st.selectbox(
                "Select Regression Model:",
                ["Linear Regression", "Random Forest Regressor"]
            )
        
        # Hyperparameter tuning
        st.markdown("<h3>Hyperparameters</h3>", unsafe_allow_html=True)
        
        if model_name == "Logistic Regression":
            c_value = st.slider("C (Regularization strength):", 0.01, 10.0, 1.0)
            max_iter = st.slider("Maximum Iterations:", 100, 1000, 100)
            solver = st.selectbox("Solver:", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"])
            
            model_params = {
                "C": c_value,
                "max_iter": max_iter,
                "solver": solver
            }
            
        elif model_name == "Random Forest" or model_name == "Random Forest Regressor":
            n_estimators = st.slider("Number of Trees:", 10, 500, 100)
            max_depth = st.slider("Maximum Depth:", 1, 50, 10)
            min_samples_split = st.slider("Minimum Samples to Split:", 2, 20, 2)
            
            model_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split
            }
            
        elif model_name == "Support Vector Machine":
            c_value = st.slider("C (Regularization strength):", 0.01, 10.0, 1.0)
            kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
            
            model_params = {
                "C": c_value,
                "kernel": kernel
            }
            
        elif model_name == "Linear Regression":
            model_params = {}  # Linear regression has no hyperparameters to tune
        
        # Train model button
        if st.button("Train Model"):
            if not feature_columns:
                st.error("Please select at least one feature column.")
            else:
                try:
                    with st.spinner("Training model..."):
                        # Preprocess data
                        X, y = preprocess_data(data, target_column, feature_columns)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                        
                        # Initialize and train model
                        if model_name == "Logistic Regression":
                            model = LogisticRegression(**model_params)
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier(**model_params)
                        elif model_name == "Support Vector Machine":
                            model = SVC(**model_params, probability=True)
                        elif model_name == "Linear Regression":
                            model = LinearRegression(**model_params)
                        elif model_name == "Random Forest Regressor":
                            model = RandomForestRegressor(**model_params)
                        
                        model.fit(X_train, y_train)
                        st.session_state.model = model
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        
                        # Evaluate model
                        st.markdown("<h3>Model Evaluation</h3>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                            
                            if task_type == "Classification":
                                accuracy = accuracy_score(y_test, y_pred)
                                st.markdown(f"<h4>Accuracy: {accuracy:.4f}</h4>", unsafe_allow_html=True)
                                
                                # Display confusion matrix
                                st.markdown("<h4>Confusion Matrix:</h4>", unsafe_allow_html=True)
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                plt.xlabel('Predicted')
                                plt.ylabel('Actual')
                                plt.title('Confusion Matrix')
                                st.pyplot(fig)
                                
                                # Display classification report
                                st.markdown("<h4>Classification Report:</h4>", unsafe_allow_html=True)
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df)
                                
                            else:  # Regression
                                mse = mean_squared_error(y_test, y_pred)
                                rmse = np.sqrt(mse)
                                r2 = r2_score(y_test, y_pred)
                                
                                st.markdown(f"<h4>Mean Squared Error: {mse:.4f}</h4>", unsafe_allow_html=True)
                                st.markdown(f"<h4>Root Mean Squared Error: {rmse:.4f}</h4>", unsafe_allow_html=True)
                                st.markdown(f"<h4>R² Score: {r2:.4f}</h4>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        with col2:
                            # Feature importance
                            if model_name in ["Random Forest", "Random Forest Regressor"]:
                                st.markdown("<h4>Feature Importance</h4>", unsafe_allow_html=True)
                                
                                importances = model.feature_importances_
                                indices = np.argsort(importances)[::-1]
                                
                                feature_importance_df = pd.DataFrame({
                                    'Feature': [feature_columns[i] for i in indices],
                                    'Importance': importances[indices]
                                })
                                
                                fig = px.bar(
                                    feature_importance_df,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title='Feature Importance'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Actual vs Predicted plot for regression
                            if task_type == "Regression":
                                st.markdown("<h4>Actual vs Predicted Values</h4>", unsafe_allow_html=True)
                                
                                pred_df = pd.DataFrame({
                                    'Actual': y_test,
                                    'Predicted': y_pred
                                })
                                
                                fig = px.scatter(
                                    pred_df,
                                    x='Actual',
                                    y='Predicted',
                                    title='Actual vs Predicted Values'
                                )
                                
                                # Add perfect prediction line
                                min_val = min(pred_df['Actual'].min(), pred_df['Predicted'].min())
                                max_val = max(pred_df['Actual'].max(), pred_df['Predicted'].max())
                                fig.add_shape(
                                    type='line',
                                    x0=min_val,
                                    y0=min_val,
                                    x1=max_val,
                                    y1=max_val,
                                    line=dict(color='red', dash='dash')
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download trained model
                        st.markdown("<h3>Download Trained Model</h3>", unsafe_allow_html=True)
                        st.markdown(get_model_download_link(model, f"{model_name.replace(' ', '_').lower()}_model.joblib"), unsafe_allow_html=True)
                        
                        st.success("Model training completed successfully!")
                        
                except Exception as e:
                    st.error(f"Error during model training: {e}")
    else:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")

# Make Predictions
elif app_mode == "Make Predictions":
    st.markdown("<h2 class='section-header'>Make Predictions</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("Please train a model first in the 'Model Training' section.")
    elif st.session_state.data is None:
        st.warning("Please upload a dataset first in the 'Upload Dataset' section.")
    else:
        data = st.session_state.data
        model = st.session_state.model
        target_column = st.session_state.target_column
        feature_columns = st.session_state.feature_columns
        model_type = st.session_state.model_type
        
        st.markdown("<h3>Input Data for Prediction</h3>", unsafe_allow_html=True)
        
        # Choose prediction method
        prediction_method = st.radio(
            "Choose prediction method:",
            ["Single Prediction", "Batch Prediction"]
        )
        
        if prediction_method == "Single Prediction":
            # Create input fields for each feature
            st.markdown("<h4>Enter values for prediction:</h4>", unsafe_allow_html=True)
            
            input_data = {}
            
            for feature in feature_columns:
                if data[feature].dtype == 'object' or data[feature].dtype.name == 'category':
                    # Categorical feature
                    unique_values = data[feature].unique().tolist()
                    input_data[feature] = st.selectbox(f"{feature}:", unique_values)
                else:
                    # Numerical feature
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    input_data[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
            
            if st.button("Make Prediction"):
                try:
                    # Create a DataFrame from input
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess input data
                    X_input = preprocess_data(input_df, None, feature_columns, is_training=False)
                    
                    # Make prediction
                    prediction = model.predict(X_input)
                    
                    st.markdown("<h3>Prediction Result</h3>", unsafe_allow_html=True)
                    
                    if model_type == "Classification":
                        # Get class probabilities if available
                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(X_input)[0]
                            
                            # Get original class labels
                            le = st.session_state.label_encoders.get(target_column)
                            if le:
                                class_names = le.classes_
                                predicted_class = le.inverse_transform([prediction[0]])[0]
                                
                                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                                st.markdown(f"<h4>Predicted Class: {predicted_class}</h4>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Display class probabilities
                                prob_df = pd.DataFrame({
                                    'Class': class_names,
                                    'Probability': probabilities
                                })
                                
                                fig = px.bar(
                                    prob_df,
                                    x='Class',
                                    y='Probability',
                                    title='Class Probabilities'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                                st.markdown(f"<h4>Predicted Class: {prediction[0]}</h4>", unsafe_allow_html=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                            st.markdown(f"<h4>Predicted Class: {prediction[0]}</h4>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:  # Regression
                        st.markdown(f"<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<h4>Predicted Value: {prediction[0]:.4f}</h4>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
        
        else:  # Batch Prediction
            st.markdown("<h4>Upload a CSV file with feature data:</h4>", unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Choose a CSV file for batch prediction", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Load prediction data
                    pred_data = pd.read_csv(uploaded_file)
                    
                    # Check if all required features are present
                    missing_features = [feature for feature in feature_columns if feature not in pred_data.columns]
                    
                    if missing_features:
                        st.error(f"Missing required features in the uploaded file: {', '.join(missing_features)}")
                    else:
                        st.markdown("<h4>Preview of uploaded data:</h4>", unsafe_allow_html=True)
                        st.dataframe(pred_data.head())
                        
                        if st.button("Make Batch Predictions"):
                            with st.spinner("Making predictions..."):
                                # Preprocess input data
                                X_input = preprocess_data(pred_data, None, feature_columns, is_training=False)
                                
                                # Make predictions
                                predictions = model.predict(X_input)
                                
                                # Add predictions to the dataframe
                                pred_data['Prediction'] = predictions
                                
                                # If classification, add class probabilities
                                if model_type == "Classification" and hasattr(model, "predict_proba"):
                                    probabilities = model.predict_proba(X_input)
                                    
                                    # Get original class labels
                                    le = st.session_state.label_encoders.get(target_column)
                                    if le:
                                        class_names = le.classes_
                                        
                                        for i, class_name in enumerate(class_names):
                                            pred_data[f'Probability_{class_name}'] = probabilities[:, i]
                                        
                                        # Convert numeric predictions back to original labels
                                        pred_data['Prediction'] = le.inverse_transform(predictions)
                                
                                st.markdown("<h3>Prediction Results</h3>", unsafe_allow_html=True)
                                st.dataframe(pred_data)
                                
                                # Download predictions
                                st.markdown("<h4>Download Predictions:</h4>", unsafe_allow_html=True)
                                st.markdown(get_csv_download_link(pred_data, "predictions.csv"), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a CSV file with feature data for batch prediction.")
