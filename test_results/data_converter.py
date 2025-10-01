import pandas as pd
file = "J:/Portofolio Diri/BRIN/aplikasi/test_results/evaluasi_deepLUNA.csv"
o_file = "evaluasi_deepLUNA.xlsx"
col_conv = ["confidence", "prob_normal", "prob_tb", "processing_time"]

df = pd.read_csv(file)
for col in col_conv:
    df[col] = pd.to_numeric(df[col])

df_new = df.to_excel(o_file, index = False)
df_new
print(f"file baru sudah dibuat dengan nama {df_new}")