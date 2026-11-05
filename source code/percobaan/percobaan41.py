# Example 2.19
# pip install xlsxwriter
import pandas

df = pandas.read_excel("test.xlsx", sheet_name="Sheet1")
print(df)