# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# def preprocess_data(file_path):
#     df = pd.read_csv(file_path)
#     df = df.dropna(subset=["Customer ID"])
#     df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
#     df["TotalSales"] = df["Quantity"] * df["Price"]
#     return df

# def calculate_rfm(df, reference_date):
#     df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
#     rfm = df.groupby("Customer ID").agg({
#         "InvoiceDate": lambda x: (reference_date - x.max()).days,  # Recency
#         "Invoice": "nunique",  # Frequency 
#         "TotalSales": "sum"  # Monetary
#     }).rename(columns={"InvoiceDate": "Recency", "Invoice": "Frequency", "TotalSales": "Monetary"})
#     return rfm

# def normalize_data(rfm):
#     scaler = StandardScaler()
#     return scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
