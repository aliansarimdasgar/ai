from crewai.tools import tool
import pandas as pd
import os
from transformers import pipeline


@tool("CSV Metadata Reader")
def csv_metadata_reader(query: str) -> str:
    """Reads metadata from all CSV files in a given directory."""
    directory_path=os.getenv('DATA_FOLDER')
    try:
        if not os.path.isdir(directory_path):
            return f"Error: {directory_path} is not a valid directory."

        metadata_summary = {}
        # Convert query to lowercase for better matching
        query = query.lower()
        # classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",multi_label=True)

        # Determine the operation based on keywords in the query
        labels = ["mismatch", "s1_duplicates", "s2_duplicates", "s1_not_in_s2","s2_not_in_s1"]
        files_label = classifier(query, labels, multi_label=False)

        # Get the top intent and its confidence score
        top_intent, top_score = files_label["labels"][0], files_label["scores"][0]

        # If the confidence is below the threshold, return fallback
        file_keyword= top_intent if top_score >= 0.3 else "No relevant intent detected"
        if file_keyword != "No relevant intent detected":
            csv_file = [f for f in os.listdir(directory_path) if f.endswith(".csv") and file_keyword in f.lower()]
            if csv_file:
                filename=csv_file[0]
                file_path = os.path.join(directory_path, filename)
                df = pd.read_csv(file_path)
                # Perform the selected operation
                if "mismatch" in file_keyword:
                    metadata_summary[filename] = {
                    "columns_name": df.columns.tolist(),
                    "row_count": df.shape[0],
                    "column_count": df.shape[1]
                    }
                elif "duplicates" in file_keyword:
                    metadata_summary[filename] = {
                    "columns_name": df.columns.tolist(),
                    "row_count": df.shape[0],
                    "column_count": df.shape[1]
                }
                elif "_not_in_" in file_keyword:
                    metadata_summary[filename] = {
                    "columns_name": df.columns.tolist(),
                    "row_count": df.shape[0],
                    "column_count": df.shape[1]
                }


        # for filename in os.listdir(directory_path):
        #     if filename.endswith(".csv"):
        #         file_path = os.path.join(directory_path, filename)
        #         df = pd.read_csv(file_path)
        #         metadata_summary[filename] = {
        #             "columns": df.columns.tolist(),
        #             "row_count": df.shape[0],
        #             "column_count": df.shape[1]
        #         }

        return str(metadata_summary) if metadata_summary else "No CSV files found in the directory."
    except Exception as e:
        return f"Error reading CSV metadata: {str(e)}"



@tool("CSV Query Runner")
def csv_query_runner(query: str):
    """
    Analyzes a user's query using NLP-based keyword detection and executes the relevant CSV analysis.

    :param query: The user's natural language query.
    :return: The results of the executed query.
    """
    directory_path=os.getenv('DATA_FOLDER')
    try:
        if not os.path.isdir(directory_path):
            return f"Error: {directory_path} is not a valid directory."

        results = {}

        # Convert query to lowercase for better matching
        query = query.lower()
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        # Determine the operation based on keywords in the query
        labels = ["mismatch", "s1_duplicates", "s2_duplicates", "s1_not_in_s2","s2_not_in_s1"]
        files_label = classifier(query, labels, multi_label=False)

        # Get the top intent and its confidence score
        top_intent, top_score = files_label["labels"][0], files_label["scores"][0]

        # If the confidence is below the threshold, return fallback
        file_keyword= top_intent if top_score >= 0.3 else "No relevant intent detected"

        if file_keyword != "No relevant intent detected":
            csv_file = [f for f in os.listdir(directory_path) if f.endswith(".csv") and file_keyword in f.lower()]
            if csv_file:
                filename=csv_file[0]
                file_path = os.path.join(directory_path, filename)
                df = pd.read_csv(file_path)

                # Perform the selected operation
                if "mismatch" in file_keyword:
                    results[filename] = {
                    "all_mismatch_data":find_all_mismatches(df),
                    # "columns": df.columns.tolist(),
                    # "row_count": df.shape[0],
                    # "column_count": df.shape[1]
                    }
                elif "duplicates" in file_keyword:
                    results[filename] = {
                    "all_duplicate_data":df,
                    # "columns": df.columns.tolist(),
                    # "row_count": df.shape[0],
                    # "column_count": df.shape[1]
                }
                elif "_not_in_" in file_keyword:
                    results[filename] = {
                    "all_data":df,
                    # "columns": df.columns.tolist(),
                    # "row_count": df.shape[0],
                    # "column_count": df.shape[1]
                }
                else:
                    results[filename] = df.to_dict()  

        return results if results else "No matching data found for your request."

    except Exception as e:
        return f"Error processing CSV files: {str(e)}"


import pandas as pd

def find_all_mismatches(df):
    """
    Identifies mismatches across all source-target column pairs in a DataFrame.
    Returns an LLM-friendly JSON-like dictionary with mismatched row indices.

    :param df: DataFrame where source and target columns are paired consecutively
    :return: Dictionary containing mismatched column pairs and their details.
    """
    if df.shape[1] % 2 != 0:
        raise ValueError("The DataFrame must have an even number of columns for proper pairing.")

    df_copy = df.astype(str)  # Convert all columns to string for uniform comparison
    mismatch_dict = {}

    # Iterate through column pairs
    for i in range(0, df_copy.shape[1], 2):  
        col_source, col_target = df_copy.columns[i], df_copy.columns[i + 1]

        # Find mismatches using .ne() (not equal)
        mismatch_mask = df_copy[col_source].ne(df_copy[col_target])

        if mismatch_mask.any():
            mismatch_dict[f"{col_source}_vs_{col_target}"] = [
                {"row": idx, "source": df_copy.at[idx, col_source], "target": df_copy.at[idx, col_target]}
                for idx in mismatch_mask[mismatch_mask].index
            ]

    return mismatch_dict



# def find_all_mismatches(df):
#     """
#     Identifies mismatches across all source-target column pairs in a DataFrame.
#     Assumes that each consecutive pair of columns (1st & 2nd, 3rd & 4th, etc.) should match.

#     :param df: DataFrame where source and target columns are paired consecutively
#     :return: DataFrame with mismatched rows and details on mismatched columns
#     """
#     df_copy = df.astype(str)  # Convert all columns to string
#     mismatch_details = []

#     # Iterate through column pairs
#     for i in range(0, df_copy.shape[1] - 1, 2):  # Step by 2 to check pairs
#         col_source = df_copy.columns[i]
#         col_target = df_copy.columns[i + 1]

#         # Find mismatches
#         current_mismatch = df_copy[col_source] != df_copy[col_target]
#         mismatch_details.append(current_mismatch.rename(f"Mismatch_{col_source}_vs_{col_target}"))

#     # If no mismatches, return empty DataFrame
#     if not mismatch_details:
#         return pd.DataFrame(columns=df.columns)

#     # Combine all mismatch details
#     mismatch_df = df.loc[pd.concat(mismatch_details, axis=1).any(axis=1)].copy()
#     for detail in mismatch_details:
#         mismatch_df[detail.name] = detail

#     return mismatch_df




# def find_all_mismatches(df):
#     """
#     Identifies mismatches across all source-target column pairs in a DataFrame.
#     Assumes that each consecutive pair of columns (1st & 2nd, 3rd & 4th, etc.) should match.
    
#     :param df: DataFrame where source and target columns are paired consecutively
#     :return: DataFrame with mismatched rows and details on mismatched columns
#     """
#     mismatch_mask = pd.Series(False, index=df.index)  # Initialize a mask to track mismatches
#     mismatch_details = []

#     # Iterate through column pairs
#     for i in range(0, df.shape[1] - 1, 2):  # Step by 2 to check pairs
#         col_source = df.columns[i]
#         col_target = df.columns[i + 1]

#         # Standardize date formats (if applicable)
#         if df[col_source].dtype == "object" and df[col_target].dtype == "object":
#             try:
#                 df[col_source] = pd.to_datetime(df[col_source], errors="coerce").astype(str)
#                 df[col_target] = pd.to_datetime(df[col_target], errors="coerce").astype(str)
#             except Exception:
#                 pass  # Ignore errors if conversion is not possible

#         # Find mismatches
#         current_mismatch = df[col_source] != df[col_target]
#         mismatch_mask |= current_mismatch

#         # Store mismatch details
#         mismatch_details.append(current_mismatch.rename(f"Mismatch_{col_source}_vs_{col_target}"))

#     # Combine all mismatch details into the DataFrame
#     mismatch_df = df[mismatch_mask].copy()
#     for detail in mismatch_details:
#         mismatch_df[detail.name] = detail

#     return mismatch_df





# @tool("CSV Query Runner")
# def csv_query_runner(directory_path: str, query_name: str) -> str:
#     """Runs predefined queries on all CSV files in the specified directory."""
#     directory_path=os.getenv('DATA_FOLDER')
#     try:
#         if not os.path.isdir(directory_path):
#             return f"Error: {directory_path} is not a valid directory."

#         predefined_queries = {
#             "count_rows": lambda df: df.shape[0],
#             "list_columns": lambda df: df.columns.tolist(),
#             "null_counts": lambda df: df.isnull().sum().to_dict(),
#             "summary_statistics": lambda df: df.describe().to_dict(),
#         }

#         if query_name not in predefined_queries:
#             return f"Error: Query '{query_name}' is not predefined."

#         query_func = predefined_queries[query_name]
#         query_results = {}

#         for filename in os.listdir(directory_path):
#             if filename.endswith(".csv"):
#                 file_path = os.path.join(directory_path, filename)
#                 df = pd.read_csv(file_path)
#                 query_results[filename] = query_func(df)

#         return str(query_results) if query_results else "No CSV files found in the directory."
#     except Exception as e:
#         return f"Error running CSV query: {str(e)}"



# class CSVMetadataInput(BaseModel):
#     """Input schema for CSVMetadataTool."""
#     file_path: str = Field(..., description="Path to the CSV file")

# @tool("CSV Metadata Reader")
# def csv_metadata_reader(file_path: str) -> str:
#     """Reads complete metadata from a CSV file, including column names, data types, row count and file size."""
#     try:
#         df = pd.read_csv(file_path)
#         metadata = {
#             "columns": df.columns.tolist(),
#             "data_types": df.dtypes.astype(str).to_dict(),
#             "row_count": df.shape[0],
#             "column_count": df.shape[1],
#             "memory_usage": df.memory_usage(deep=True).sum()
#         }
#         return str(metadata)
#     except Exception as e:
#         return f"Error reading CSV metadata: {str(e)}"
        





# @tool("CSV Query Runner")
# def csv_query_runner(query: str):
#     """
#     Analyzes a user's query using NLP-based keyword detection and executes the relevant CSV analysis.

#     :param query: The user's natural language query.
#     :return: The results of the executed query.
#     """
#     directory_path=os.getenv('DATA_FOLDER')
#     try:
#         if not os.path.isdir(directory_path):
#             return f"Error: {directory_path} is not a valid directory."

#         results = {}

#         # Convert query to lowercase for better matching
#         query = query.lower()

#         # Determine the operation based on keywords in the query
#         mismatch_keywords = [
#                     "mismatch","difference", "discrepancy", "inconsistency", "variance", "deviation", "conflict",
#                     "error", "divergence", "inequality", "incongruence", "discordance", "difference",
#                     "variation", "distinction", "contrast", "change", "gap", "differentiation",
#                     "does not match", "unequal", "not the same", "differing", "does not align",
#                     "unmatched", "unaligned"
#                 ]
        
#         duplicate_keywords= [
#                     "duplicate", "duplicates", "repeated", "repeat", "redundant", "replicated", 
#                     "same record", "identical", "copy", "cloned", "matching records", 
#                     "twin records", "exact match", "mirror entry", "duplicated data", "reoccurring",
#                     "recurring", "reappearance", "carbon copy", "copy-paste", "non-unique", 
#                     "replication", "reproduction", "identical entry", "same data", "extra entry"
#                 ]
#         if any(word in query for word in mismatch_keywords):
#             file_keyword = "mismatch"
#         elif any(word in query for word in duplicate_keywords):
#             if any(word in query for word in ["s1","source","first"]):
#                 file_keyword = "s1_duplicates"
#             elif any(word in query for word in ["s2","target","second"]):
#                 file_keyword = "s2_duplicates"
#             else:
#                 file_keyword = "duplicates"

#         elif any(word in query for word in ["missing", "not in s2", "not in second"]):
#             file_keyword = "s1_not_in_s2"

#         elif any(word in query for word in ["not in s1", "not in first"]):
#             file_keyword = "s2_not_in_s1"

#         else:
#             return "Sorry, I couldn't determine the request. Please rephrase."

#         # Search for the corresponding file
#         for filename in os.listdir(directory_path):
#             if filename.endswith(".csv") and file_keyword in filename.lower():
#                 file_path = os.path.join(directory_path, filename)
#                 df = pd.read_csv(file_path)

#                 # Perform the selected operation
#                 if "mismatch" in file_keyword:
#                     results[filename] = {
#                     "all_mismatch_data":find_all_mismatches(df),
#                     "columns": df.columns.tolist(),
#                     "row_count": df.shape[0],
#                     "column_count": df.shape[1]
#                 }
#                 elif "duplicate" in file_keyword:
#                     results[filename] = {
#                     "all_duplicate_data":df,
#                     "columns": df.columns.tolist(),
#                     "row_count": df.shape[0],
#                     "column_count": df.shape[1]
#                 }
#                 else:
#                     results[filename] = df  # For missing records

#         return results if results else "No matching data found for your request."

#     except Exception as e:
#         return f"Error processing CSV files: {str(e)}"

# def find_all_mismatches(df):
#     """
#     Identifies mismatches across all source-target column pairs in a DataFrame.
#     Assumes that each consecutive pair of columns (1st & 2nd, 3rd & 4th, etc.) should match.
    
#     :param df: DataFrame where source and target columns are paired consecutively
#     :return: DataFrame with mismatched rows and details on mismatched columns
#     """
#     mismatch_mask = pd.Series(False, index=df.index)  # Initialize a mask to track mismatches
#     mismatch_details = []

#     # Iterate through column pairs
#     for i in range(0, df.shape[1] - 1, 2):  # Step by 2 to check pairs
#         col_source = df.columns[i]
#         col_target = df.columns[i + 1]

#         # Standardize date formats (if applicable)
#         if df[col_source].dtype == "object" and df[col_target].dtype == "object":
#             try:
#                 df[col_source] = pd.to_datetime(df[col_source], errors="coerce").astype(str)
#                 df[col_target] = pd.to_datetime(df[col_target], errors="coerce").astype(str)
#             except Exception:
#                 pass  # Ignore errors if conversion is not possible

#         # Find mismatches
#         current_mismatch = df[col_source] != df[col_target]
#         mismatch_mask |= current_mismatch

#         # Store mismatch details
#         mismatch_details.append(current_mismatch.rename(f"Mismatch_{col_source}_vs_{col_target}"))

#     # Combine all mismatch details into the DataFrame
#     mismatch_df = df[mismatch_mask].copy()
#     for detail in mismatch_details:
#         mismatch_df[detail.name] = detail

#     return mismatch_df



# @tool("CSV Query Runner")
# def csv_query_runner() -> str:
#     """Runs predefined queries on all CSV files in the specified directory."""
#     directory_path=os.getenv('DATA_FOLDER')
#     try:
#         if not os.path.isdir(directory_path):
#             return f"Error: {directory_path} is not a valid directory."

#         metadata_summary = {}

#         for filename in os.listdir(directory_path):
#             if filename.endswith(".csv"):
#                 file_path = os.path.join(directory_path, filename)
#                 df = pd.read_csv(file_path)

#                 metadata_summary[filename] = {
#                     "all_mismatch_data": find_all_mismatches(df),
#                     "count_rows": lambda df: df.shape[0],
#                     "list_columns": lambda df: df.columns.tolist(),
#                     "null_counts": lambda df: df.isnull().sum().to_dict(),
#                     "summary_statistics": lambda df: df.describe().to_dict(),
#                 }

#         return str(metadata_summary) if metadata_summary else "No CSV files found in the directory."
#     except Exception as e:
#         return f"Error reading CSV metadata: {str(e)}"