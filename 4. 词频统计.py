# 词频统计
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


def cntv_func(click_log, columns_dict={}):
    '''
    输入:click_log     csv  训练集测试集总的点击日志
         columns_dict  dict 需要统计的特征名字和所在的索引{'time':0, 'creative_id':2}
    输出:直接保存为稀疏向量
    '''
    for key in columns_dict:
        print('current key is : %s' % key)
        user_click_history = defaultdict(str)
        for single_click_log in tqdm(click_log.values, total=click_log.shape[0]):
            if user_click_history[single_click_log[1]] == '':
            user_click_history[single_click_log[1]] = ' '.join(
                [str(single_click_log[columns_dict[key]]) for i in range(int(single_click_log[3]))])
            else:
            user_click_history[single_click_log[1]] = user_click_history[single_click_log[1]] + ' ' + \
                ' '.join([str(single_click_log[columns_dict[key]])
                          for i in range(int(single_click_log[3]))])

        all_user_click_history = []
        for k in range(1, 900001):
            all_user_click_history.append(user_click_history[k])
        for k in range(3000001, 4000001):
            all_user_click_history.append(user_click_history[k])
        #点击历史需要空格分隔开['12 23 32', '21 12']

        cntv = CountVectorizer(min_df=30, dtype=np.int16,
                               token_pattern=r"(?u)\b\w+\b", max_features=10000)
        #token_pattern使用正则表达式来匹配字符默认"(?u)\b\ww+\b"忽略单字符
        cntv_user = cntv.fit_transform(all_user_click_history)
        if key == 'time':
            all_cntv_user = cntv_user
        else:
            all_cntv_user = sparse.hstack((all_cntv_user, cntv_user))
        print(cntv_user.shape)
    sparse.save_npz('./data/cntv.npz', all_cntv_user)
