# 用户聚类统计特征
# 基于用户的点击历史对用户塑造大概的轮廓

agg_func = {
    'creative_id_age_1_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_2_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_3_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_4_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_5_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_6_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_7_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_8_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_9_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_age_10_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_gender_1_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id_gender_2_kfold_mean':  ['mean', 'max', 'min', 'std', 'median'],
    'creative_id':  ['count', 'nunique'],
    'click_times':  ['sum', 'mean'],
    'time':  ['nunique'],
}

train_user_aggregate_feature = train_click_log.groupby(
    ['user_id']).agg(agg_func)
test_user_aggregate_feature = test_click_log.groupby(['user_id']).agg(agg_func)
