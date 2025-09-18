import pandas as pd
import os

def sample_data():

    """Loads a sample retail store sales dataset for demonstration and testing.

    The dataset is a small CSV file included with the library. It is useful for 
    quickly experimenting with the data cleaning and validation functions without 
    needing an external data source.

    Returns:
        pd.DataFrame: A pandas DataFrame containing sample retail sales data.

    Example:
        >>> from cleanframe.data import sample_data
        >>>
        >>> df = sample_data()
        >>> print(df.head())
          transaction_id customer_id       category           item  price_per_unit  quantity  total_spent payment_method  location transaction_date  discount_applied
        0  TXN_6867343     CUST_09  Patisserie   Item_10_PAT            18.5      10.0        185.0   Digital Wallet    Online       2024-04-08             True
        1  TXN_3731986     CUST_22  Milk Products  Item_17_MILK            29.0       9.0        261.0   Digital Wallet    Online       2023-07-23             True
        2  TXN_9303719     CUST_02      Butchers   Item_12_BUT            21.5       2.0         43.0     Credit Card    Online       2022-10-05            False
        3  TXN_9458126     CUST_06    Beverages   Item_16_BEV            27.5       9.0        247.5     Credit Card    Online       2022-05-07              NaN
        4  TXN_4575373     CUST_05         Food   Item_6_FOOD            12.5       7.0         87.5   Digital Wallet    Online       2022-10-02            False
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.join(current_dir, "sample_data", "retail_store_sales.csv")

    return pd.read_csv(file_path)