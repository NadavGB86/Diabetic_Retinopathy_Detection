from imutils import paths
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", 15)
pd.set_option("display.max_rows", 100)
pd.set_option("expand_frame_repr", False)


def get_datasets(frac=0.5):
    df = pd.DataFrame(list(paths.list_images("data\\train_images")), columns=['File Path'])
    df['Label'] = df['File Path'].map(lambda par: par.split('class')[1].replace('.jpg', ''))
    df['idx'] = df['File Path'].map(lambda par: par.split('_')[3].replace('idx', '')).astype('int')
    df['File Name'] = df['File Path'].map(lambda par: par.split('\\')[2].replace('.jpg', ''))
    df['Patient Id'] = df['File Name'].map(lambda par: par.split('_')[0])
    df['side'] = df['File Name'].map(lambda par: par.split('_')[1])

    retina_df = df[(df['Label'] == '0') | \
                   ((df['Label'] == '1') & (df['idx'] < 3)) | \
                   ((df['Label'] == '2') & (df['idx'] < 2)) | \
                   ((df['Label'] == '3') & (df['idx'] < 5)) | \
                   ((df['Label'] == '4') & (df['idx'] < 5))]

    # print(retina_df)

    rr_df = retina_df[['Patient Id', 'Label']].drop_duplicates()
    train_ids, valid_ids = train_test_split(rr_df['Patient Id'],
                                            test_size=0.25,
                                            random_state=2018,
                                            stratify=rr_df['Label'])
    raw_train_df = retina_df[retina_df['Patient Id'].isin(train_ids)]
    raw_valid_df = retina_df[retina_df['Patient Id'].isin(valid_ids)]
    raw_valid_df = raw_valid_df[raw_valid_df['idx'] == 0]
    # print('train', raw_train_df.shape[0], 'validation', raw_valid_df.shape[0])

    train_df = raw_train_df.groupby(['Label', 'side']).sample(frac=frac, replace=False).reset_index(drop=True)
    # print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])

    new_valid_ids, test_ids = train_test_split(raw_valid_df['Patient Id'],  # to keep track on new df grouped PatientId
                                               test_size=0.25,
                                               random_state=2018,
                                               shuffle=True,
                                               stratify=raw_valid_df['Label'])
    # now make new df
    valid_df = raw_valid_df[raw_valid_df['Patient Id'].isin(new_valid_ids)].reset_index(drop=True)
    test_df = raw_valid_df[raw_valid_df['Patient Id'].isin(test_ids)].reset_index(drop=True)

    # print(f'new_validation: {valid_df.shape[0]}, test: {test_df.shape[0]}')
    #
    # print(train_df.shape)
    # print(valid_df.shape)
    # print(test_df.shape)

    return train_df, valid_df, test_df

