# w2v fasttext deepwalk

# w2v
from gensim.models import Word2Vec

sentences = click_log.groupby(['user_id', 'time']).apply(
    lambda x: x['creative_id'].tolist()).tolist()
w2v_size = 32
model = Word2Vec(sentences=sentences, min_count=1, sg=1,
                 window=8, size=w2v_size, workers=-1, seed=2020, iter=10)
train_X = np.array(click_log.groupby(['user_id']).apply(
    lambda x: model[x['creative_id']].mean(axis=0)))

# fasttext
import fasttext
def train_embedding_model(df, id_name):
    model_path = base_dir + "model/{}_{}_sg.bin".format(id_name, vector_dim)
    if not os.path.exists(model_path):
        print("train {}_{} embedding model!".format(id_name, vector_dim))
        df[id_name] = df[id_name].astype("str")
        user_time_group_df = df.groupby(["user_id", "time"])[
            id_name].agg(" ".join).reset_index()
        user_group_df = user_time_group_df.groupby(
            "user_id")[id_name].agg(" ".join).reset_index()
        id_csv = base_dir + "embedding/{}.csv".format(id_name)
        user_group_df[id_name].to_csv(id_csv, index=False, header=False)
        fasttext_model = fasttext.train_unsupervised(
            id_csv, model="skipgram", minn=0, maxn=0, ws=50, dim=vector_dim)
        fasttext_model.save_model(model_path)
    else:
        print("{}_{} embedding model already exist!".format(id_name, vector_dim))


target_id_list = ['creative_id', 'ad_id', 'advertiser_id']
for target_id in target_id_list:
    print("start {} embedding!".format(target_id))
    train_embedding_model(all_data_df, target_id)

# deepwalk
def deepwalk(f1, f2, flag, L):
    # deepwalk会将主键和要训练的键的词向量都训练出来
    path3 = "data/log.pkl"
    log = pd.read_pickle(path3)
    log = log[[f1, f2]]
    gc.collect()
    print("data load!")
    # Deepwalk算法，
    print("deepwalk:", f1, f2)
    # 构建图
    dic = {}
    for item in log[[f1, f2]].values:
        try:
            str(int(item[1]))
            str(int(item[0]))
        except:
            continue
        try:
            dic['item_'+str(int(item[1]))].add('user_'+str(int(item[0])))
        except:
            dic['item_'+str(int(item[1]))] = set(['user_'+str(int(item[0]))])
        try:
            dic['user_'+str(int(item[0]))].add('item_'+str(int(item[1])))
        except:
            dic['user_'+str(int(item[0]))] = set(['item_'+str(int(item[1]))])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])
        dic_cont[key] = len(dic[key])
    print("creating")
    # 构建路径
    path_length = 10
    sentences = []
    length = []
    for key in dic:
        sentence = [key]
        while len(sentence) != path_length:
            key = dic[sentence[-1]
                      ][random.randint(0, dic_cont[sentence[-1]]-1)]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 1000000 == 0:
            print(len(sentences))

    print(np.mean(length))
    print(len(sentences))
    del dic
    del length
    gc.collect()

    # 训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4,
                     min_count=1, sg=1, workers=-1, iter=20)
    print('outputing...')
    # 输出
    # values=set(log[f1].values)
    # w2v=[]
    # for v in values:
    #    try:
    #        a=[int(v)]
    #        a.extend(model['user_'+str(int(v))])
    #        w2v.append(a)
    #    except:
    #        pass
    # out_df=pd.DataFrame(w2v)
    # names=[f1]
    # for i in range(L):
    #    names.append(f1+'_'+ f2+'_'+names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    #out_df.columns = names
    # print(out_df.head())
    #out_df.to_pickle('data/' +f1+'_'+ f2+'_'+f1 +'_'+flag +'_deepwalk_'+str(L)+'.pkl')

    del sentences
    gc.collect()

    ########################
    values = set(log[f2].values)
    w2v = []
    for v in values:
        try:
            a = [int(v)]
            a.extend(model['item_'+str(int(v))])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [f2]
    for i in range(L):
        names.append(f1+'_' + f2+'_' +
                     names[0]+'_deepwalk_embedding_'+str(L)+'_'+str(i))
    out_df.columns = names
    # print(out_df.head())
    out_df.to_pickle('data/' + f1+'_' + f2+'_'+f2 + '_' +
                     flag + '_deepwalk_'+str(L)+'.pkl')
