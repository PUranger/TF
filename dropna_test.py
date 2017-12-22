import pandas as pd
import numpy as np

#df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3,4,np.nan,1],
#                  [np.nan, np.nan, np.nan, 5]],
#                  columns = list('ABCD'))

df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3,4,np.nan,1],  #example for only 1 column have non nan

                  [np.nan, 1, np.nan, 5]],
                  columns = list('ABCD'))

#df = pd.DataFrame([[1, 2, 1, 0], [3,4,np.nan,1],    #example for only 1 row have non nan
#                  [np.nan, np.nan, np.nan, 5]],
#                  columns = list('ABCD'))

df.dropna(inplace=True)
print(df)


#In conclusion, df.dropna(inplace=True) would delete all rows that contains NaN
