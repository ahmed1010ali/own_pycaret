#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3

from pycaret.classification import setup, compare_models
from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg

def load_and_preprocess_data():
    while True:
        try:
            data_format = input("Enter the data format (CSV, Excel, SQL): ")
            
            if data_format == "csv":
                file_path = input("Enter the path to your CSV file: ")
                df = pd.read_csv(file_path)
            elif data_format == "excel":
                file_path = input("Enter the path to your Excel file: ")
                df = pd.read_excel(file_path)
            elif data_format == "sql":
                database_path = input("Enter the path to your SQLite database: ")
                query = input("Enter your SQL query: ")
                conn = sqlite3.connect(database_path)
                df = pd.read_sql(query, conn)
                conn.close()
            else:
                print("Invalid data format selected.")
                continue
            
            # Print the loaded dataframe
            print(df)
            
            # Select the target column
            while True:
                chosen_target = input("Choose the Target Column: ")
                if chosen_target in df.columns:
                    break
                else:
                    print(f"Column '{chosen_target}' not found in the dataset.")
            
            # Detect the task type (regression or classification)
            if np.issubdtype(df[chosen_target].dtype, np.number):
                task_type = "Regression"
            else:
                task_type = "Classification"
            
            print(f"Task Type: {task_type}")
            
            # Handle missing values
            for column in df.columns:
                if column != chosen_target:
                    if np.issubdtype(df[column].dtype, np.number):
                        impute_method = input(f"Select imputation method for {column} (mean/median/mode): ")
                        if impute_method == "mean":
                            df[column].fillna(df[column].mean(), inplace=True)
                        elif impute_method == "median":
                            df[column].fillna(df[column].median(), inplace=True)
                        else:
                            df[column].fillna(df[column].mode()[0], inplace=True)
                    else:
                        impute_method = input(f"Select imputation method for {column} (most frequent/additional class): ")
                        if impute_method == "most frequent":
                            df[column].fillna(df[column].value_counts().idxmax(), inplace=True)
                        else:
                            df[column].fillna("Missing", inplace=True)
            
            # Select and drop columns
            while True:
                columns_to_drop = input("Select columns to drop (comma-separated): ").split(",")
                columns_to_drop = [col.strip() for col in columns_to_drop]
                valid_columns = [col for col in columns_to_drop if col in df.columns]
                
                if valid_columns:
                    df.drop(columns=valid_columns, inplace=True)
                    break
                else:
                    print("No valid columns selected for dropping.")
            
            # Use PyCaret for model selection and evaluation
            if task_type == "Classification":
                setup_df = setup(data=df, target=chosen_target, silent=True)
                best_model = compare_models()
                print(best_model)
            else:
                setup_df = setup_reg(data=df, target=chosen_target, silent=True)
                best_model_reg = compare_models_reg()
                print(best_model_reg)
            
            # You can add code for saving or downloading the model or results if needed.
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# Call the function to start the process
load_and_preprocess_data()


# In[ ]:





# In[ ]:




