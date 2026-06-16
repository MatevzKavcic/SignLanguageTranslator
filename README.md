 
1.   1ExtractFromVideos.py (extract and get a csv File out). cleanupCrew/zCrew/full_dataset_z_Primo.csv

2.   2MakeItUtf8.py .... make the characters and stuff like č ž š so it fits the format... slovenian is a good language

3.   3CleanTheData.py .... Remowe the 0 in the training so it make the model better... interpolate between the pointts that are missing 

4.   4Augmenting.py  .... augmenting videos and making 4 other version of the video... also puting it in .npy format for storage space and fast response

5.   5TrainingTesting.py  ..... traningn on the data

6.   6VideoPredistioning.py  ....  Predicting based on the model it spits out in the training.





HelperScirpts
0.   Calmate.py (Poglej ce mas prave librarije settane up)

1.     1ConcatCSVFiles   concating 2 csv files if necesary

2.1    2MakeItSuitableForTraining .... to deluje samo ce imas ze dosti videotov enega in istega in ne rabis AUGMENTATION


0.0    0.0ExtractingSpecificPoints.py   .... will be needing this for extracting specific points not all 1500

5.    5paddingIt.py ....  padding the data so it fits the format.. usles but its here its a wast to trow away

