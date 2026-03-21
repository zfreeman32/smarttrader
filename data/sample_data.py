#%%
# File paths
import pandas as pd

# File paths
input_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\datasets\EURUSD_full_1hour.txt"
output_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\trade_bot\data\currency_data\EURUSD_1hr.csv"
num_lines = 2_500_000  # Number of lines to sample

# Define column names
column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]

# Read file efficiently using tail() from pandas
df = pd.read_csv(input_file, engine="python", names=column_names, header=None)

# Take only the last N rows
df = df.tail(num_lines)

# Display DataFrame
df.head()


#%%
# Save as CSV
df.to_csv(output_file, index=False)
print(f"CSV file saved as {output_file}")

# %%
