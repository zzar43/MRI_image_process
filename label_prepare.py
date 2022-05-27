import os
import pandas as pd
from tqdm import tqdm

BASE_PATH = "/Users/lida/Local_Document/ADNI_dataset/UNet_IDP_Stage_1"
table_name = "UNet_IDP_Stage_1_5_12_2022.csv"
new_table_name = "UNet_IDP_Stage_1.csv"

tf = pd.read_csv(os.path.join(BASE_PATH, table_name))

# print(tf.head)

filename_list = []
label_list = []

for i in tqdm(range(tf.shape[0]), ncols=80):
    filename = tf['Subject'][i] + '_' + tf['Image Data ID'][i]
    if tf['Group'][i] == 'EMCI':
        label = 1
    elif tf['Group'][i] == 'LMCI':
        label = 2
    elif tf['Group'][i] == 'AD':
        label = 3
    elif tf['Group'][i] == 'CN':
        label = 0
    filename_list.append(filename)
    label_list.append(label)

test_data_class = {
    'filename' : filename_list,
    'label' : label_list,
}
new_df = pd.DataFrame(test_data_class)
# print(new_df.head())
new_df.to_csv(os.path.join(BASE_PATH,new_table_name), index=False, header=False)
print('Test table is saved.')