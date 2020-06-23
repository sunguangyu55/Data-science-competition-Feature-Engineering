# 目标编码和五折特征
def get_kfold_features(data_df_, test_df_):
    '''
    输入：  data_df_         csv  训练数据，
            test_df_         csv  测试数据
    输出：  data_df          csv  包含目标编码的训练数据，
            test_df          csv  包含目标编码的测试数据，
            kfold_features1  list 目标编码创建的特征名
    '''
    data_df = data_df_.copy()
    test_df = test_df_.copy()
    folds = KFold(n_splits=5, shuffle=True, random_state=2019)

    data_df['fold'] = 0
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(data_df, data_df)):
        data_df.loc[val_idx, 'fold'] = fold_
    data_df['fold'] = data_df['fold'].astype(np.uint8)
    kfold_features1 = []

    print('二阶交叉...')
    for feat in ['creative_id']:  # 需要目标编码的特征，还可以进行三阶以上的交叉

        nums_columns = ['age_1', 'age_2', 'age_3', 'age_4', 'age_5',
                        'age_6', 'age_7', 'age_8', 'age_9', 'age_10',
                        'gender_1', 'gender_2']  # 目标编码的目标，类别目标一般需要onehot

        for f in nums_columns:
        colname1 = feat + '_' + f + '_kfold_mean'
        print(feat, f, ' mean')
        kfold_features1.append(colname1)
        data_df[colname1] = 0
        # train
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(data_df, data_df)):
            Log_trn = data_df.iloc[trn_idx]
            # mean
            order_label = Log_trn.groupby([feat])[f].mean()
            tmp = data_df.loc[data_df.fold == fold_, [feat]]
            data_df.loc[data_df.fold == fold_,
                        colname1] = tmp[feat].map(order_label)
        data_df[colname1] = data_df[colname1].astype(np.float16)
        # test
        test_df[colname1] = 0
        order_label = data_df.groupby([feat])[f].mean()
        test_df[colname1] = test_df[feat].map(order_label)
        test_df[colname1] = test_df[colname1].astype(np.float16)
        gc.collect()
    del data_df['fold']
    return data_df, test_df, kfold_features1


train_click_log, test_click_log, kfold_features1 = get_kfold_features(
    train_click_log, test_click_log)
