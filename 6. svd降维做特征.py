# svd降维做特征
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import numpy as np

all_cntv_user = sparse.load_npz('./data/cntv.npz')
tsvd = TruncatedSVD(n_components=128, random_state=2020)
cntv_svd = tsvd.fit_transform(all_cntv_user)
np.save('./data/cntv_svd.npz', cntv_svd)
