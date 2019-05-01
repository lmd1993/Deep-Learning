If you want to use L2 Norm, you need to edit the:
model.compile(optimizer='rmsprop', loss='mean_squared_error')
If you want to use Log Norm, you need:
model.compile(optimizer='rmsprop', loss='binary_crossentropy')



Log Norm No Negative:
  python generateStatsL2.py emb/embNeg5NoNeg outputStats/logNoNeg vocab.txt emb/embNeg5_2NoNeg 


L2 Norm No Negative
  python cbow_model.py 5 5 embNoNegL2 embNoNegL2_2
  python generateStatsL2.py emb/embNoNegL2 outputStats/logNoNegL2 vocab.txt emb/embNoNegL2_2 


python cbow_model.py 5 5 embNeg5 embNeg5_2
python generateStatsL2.py emb/embedding.txt outputStats/try vocab.txt emb/embedding2.txt 
