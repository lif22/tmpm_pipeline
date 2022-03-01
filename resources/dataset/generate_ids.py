#%%
import pandas as pd
import random
import os
filename = r"Mail_ApplicationDummy.xlsx"
base = r"C:\Users\flietz\OneDrive - TU Wien\!Studium\1_MSc\!Diplomarbeit\code\pipeline\resources\dataset"
sheetname = "Junior"

full = os.path.join(base, filename)
inputDf = pd.read_excel(full, sheetname)
print(len(inputDf))

for index, row in inputDf.iterrows():
    # create message id for all
    a = random.randint(100000, 999999)
    b = random.randint(100000, 999999)
    mail_last = row[0].split("@")[1]
    mail = mail_last.split(".com")[0]
    i = f"{a}.{b}@{mail}"
    inputDf.loc[index, 3] = i
    
inputDf.to_excel("Mail_ApplicationDummy_Junior_Filled.xlsx", index=False)

# %%
