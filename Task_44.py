import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)

data = pd.DataFrame({'whoAmI': lst})

encoder = OneHotEncoder()

onehot_encoded = encoder.fit_transform(data[['whoAmI']])

feature_names = encoder.get_feature_names_out(['whoAmI'])

onehot_df = pd.DataFrame(onehot_encoded.toarray(), columns=feature_names)

print(onehot_df)