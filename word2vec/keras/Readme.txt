First 5 is epochs. Second 5 is batch size. Change global setting to change negative 5 or 2 or others.
python cbow_model.py 5 5 embNeg5 embNeg5_2
python generateStatsL2.py emb/embedding.txt outputStats/try vocab.txt emb/embedding2.txt 

nohup python -u cbow_model.py 15 5 embNeg3 embNeg3_2 &
python generateStatsL2.py emb/embNeg3 outputStats/embNeg3 vocab.txt emb/embNeg3_2




L1 Norm: python cbow_model.py 10 5 embNoNegL1Epoch10 embNoNegL1Epoch10_2

