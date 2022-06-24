import pickle
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--month', type=int, default=1)
parser.add_argument('--year', type=int, default=2021)

args = parser.parse_args()

year = args.year
month = args.month

with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[4]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:


df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# In[7]:


print(np.mean(y_pred))


# In[8]:


df


# In[9]:



output_file = 'output.parquet'


# In[10]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[20]:


df_result = pd.DataFrame({'ride_id': df['ride_id'], 'pred': y_pred})


# In[21]:


df_result


# In[22]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:





# In[ ]:




