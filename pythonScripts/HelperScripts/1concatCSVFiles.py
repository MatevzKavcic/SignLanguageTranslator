import pandas as pd # type: ignore

df1 = pd.read_csv("cleanupCrew/OtherCrew/cleaned_full_datasetSecond_terco.csv")
df2 = pd.read_csv("cleanupCrew/zCrew/cleaned_full_datasetFor_z_terco.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.to_csv("FullDatasetCleaned_FINAL.csv", index=False)

print(" Combined dataset saved")


#just a script for concating nothing else, not really important, but it was easier to do it in python 