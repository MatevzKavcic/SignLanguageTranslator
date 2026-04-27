import pandas as pd

df1 = pd.read_csv("cleanupCrew/OtherCrew/cleaned_full_datasetSecond_terco.csv")
df2 = pd.read_csv("cleanupCrew/zCrew/cleaned_full_datasetFor_z_terco.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("FullDatasetCleaned_FINAL.csv", index=False)

print(" Combined dataset saved")