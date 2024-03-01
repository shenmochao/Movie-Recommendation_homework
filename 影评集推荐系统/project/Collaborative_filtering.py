import time
from scipy.sparse import csr_matrix
from Data_pre import *


# —————————————————————————————基于项的协同滤波————————————————————————————

def compute_similarity(rating_matrix):
    """
    使用皮尔逊相关系数计算项的相似度矩阵
    input:
        rating_matrix: DataFrame 评分矩阵
    return:
        similarity_matrix: DataFrame 相似度矩阵
    """
    similarity_matrix = rating_matrix.corr(method="pearson")
    return similarity_matrix


def compute_similarity_alpha(rating_matrix, alpha):
    """
    使用改进公式计算项的相似度矩阵
    input:
        rating_matrix: DataFrame 评分矩阵
        a: float 权重调节参数，0 <= a <= 1
    return:
        similarity_matrix: DataFrame 相似度矩阵
    """
    rating_matrix_sparse = csr_matrix(rating_matrix.fillna(0).values)
    item_count = np.array((rating_matrix_sparse != 0).sum(axis=0))
    denominator = np.power(item_count + 1e-8, 1 - alpha) * np.power(item_count.T + 1e-8, alpha)  # 分母
    numerator = rating_matrix_sparse.T.dot(rating_matrix_sparse)  # 分子
    similarity_matrix = pd.DataFrame(numerator / denominator, index=rating_matrix.columns,
                                     columns=rating_matrix.columns)
    return similarity_matrix


def predict_item_score_for_user(user_id, item_id, rating_matrix, similarity_matrix, k=-1):
    """
    预测用户 i 对电影 j 的评分
    input:
        user_id: Integer 用户 ID
        item_id: Integer 电影 ID
        rating_movie: DataFrame 评分矩阵
        similarity_matrix: DataFrame 相似性矩阵
        k: Integer 计算预测分数的近邻个数，默认-1，表示计算所有相似度大于0的相似项
    return:
        r: Integer 用户对该电影评分的预测值
    """

    similar_items = similarity_matrix[item_id].drop(item_id).dropna()
    similar_items = similar_items.where(similar_items > 0).dropna()
    if similar_items.empty:
        # print(f"Item {item_id} don't have similar items.")
        return None

    user_rated_items = rating_matrix.loc[user_id].dropna()
    user_rated_similar_items = similar_items.loc[list(set(similar_items.index) & set(user_rated_items.index))]
    user_rated_similar_items.sort_values(ascending=False, inplace=True)
    # print("Similar items that had rated by user:", user_rated_similar_items.shape)

    a = 0  # 相似项的评分乘以相似度的累加和
    b = 0  # 相似度的累加和
    c = 0
    for similar_item, similarity in user_rated_similar_items.iteritems():
        a += similarity * rating_matrix.loc[user_id, similar_item]
        b += similarity
        c += 1
        if c == k:
            break
    if b == 0:
        # print(f"User {user_id} don't have any rated item that similar to item {item_id}")
        return None

    r = a / b
    if r > 5.0:  # 评分预测值
        r = 5.0
    return r


def predict_all_items_score_for_user(user_id, rating_matrix, similarity_matrix, cold=10, k=-1):
    """
    跟据项的相似性矩阵预测用户 i 对非冷门未评分的项的评分
    input:
        user_id: Integer 用户 ID
        rating_movie: DataFrame 评分矩阵
        similarity_matrix: DataFrame 相似性矩阵
        cold: Integer 非冷门项要求必须存在大于等于 cold 个用户对其评分，否则判定为冷门项，不值得考虑
        k: Integer 计算预测分数的近邻个数，默认-1，表示计算所有相似度大于0的相似项
    return:
        predict_rating: {} Key: 项的ID, Value: 预测分数
    """
    # 过滤冷门电影
    score_count = rating_matrix.count()
    hot_items = score_count.where(score_count > cold).dropna()

    # 过滤已经评分的电影
    items = rating_matrix.loc[user_id]
    unrated_items = items[items.isnull()]

    predict_items = set(hot_items.index) & set(unrated_items.index)

    predict_rating = {}
    for item_id in predict_items:
        pr = predict_item_score_for_user(user_id, item_id, rating_matrix, similarity_matrix, k=k)
        if pr is not None:
            predict_rating[item_id] = pr

    return predict_rating


class CF:
    """
    初始化评分矩阵和相似度矩阵
    input:
        data_name: String 数据集名称标识
        data_file_path: String 评分数据文件路径
        step: String 评分数据文件的分隔符
        val: Boolean 是否为验证模式，验证模式下空出一块预取的评分来进行验证测试
        val_mask: ((a, b), (c, d)) 验证模式下评分矩阵空出一块的范围：a:b行，c:d列
    """

    def __init__(self, alpha, data_name, data_file_path, step=",", val=True, val_mask=((0, 100), (0, 200))):
        self.alpha = alpha
        self._matrix_path = "./Matrix/"
        self._val = val

        # 载入/计算评分矩阵
        file_name = data_name + "-rating"
        save_file_path = self._matrix_path + file_name + ".pkl"
        start = time.time()
        if os.path.exists(save_file_path):
            self._rating_matrix = load_matrix_from_pickle(save_file_path)
        else:
            self._rating_matrix = load_data_to_matrix(file_path=data_file_path, step=step)
            save_matrix_to_pickle(self._rating_matrix, self._matrix_path, file_name)
        end = time.time()
        # print("Rating Matrix:")
        # print(self._rating_matrix, "\n")
        # print("Time Cost of Loading Rating Matrix: %.2f s" % (end - start))

        # Mask
        if val:
            (u_l, u_r), (i_l, i_r) = val_mask
            self._mask_ground_truth = self._rating_matrix.iloc[u_l:u_r, i_l:i_r].copy()
            self._rating_matrix.iloc[u_l:u_r, i_l:i_r] = np.nan
            # print("Mask Ground Truth Matrix:")
            # print(self._mask_ground_truth, "\n")
            # print("Mask Rating Matrix:")
            # print(self._rating_matrix, "\n")

        # 载入/计算相似矩阵
        file_name = data_name + f"-item-similarity"
        if val:
            file_name += "-val"
        save_file_path = self._matrix_path + file_name + ".pkl"
        start = time.time()
        if os.path.exists(save_file_path):
            self._similarity_matrix = load_matrix_from_pickle(save_file_path)
        else:
            # self._similarity_matrix = compute_similarity(self._rating_matrix)
            self._similarity_matrix = compute_similarity_alpha(self._rating_matrix, self.alpha)
            save_matrix_to_pickle(self._similarity_matrix, self._matrix_path, file_name)
        end = time.time()
        # print(f"Similarity Matrix ({self._based_type}):")
        # print(self._similarity_matrix, "\n")
        # print("Time Cost of Loading Similarity Matrix: %.2f s" % (end - start))

    def top_n_recommend(self, user_id, n, k=-1):
        """
        为一个用户推荐 N 个预测评分最高的项
        input:
            user_id: Integer 用户 ID
            n: Integer 返回的推荐项的个数
            k: Integer 预测分数时考虑的相似用户个数
        return:
            predict_result: [] 每一个元素是由 Item ID 和 预测评分组成的元组
        """
        start = time.time()
        pr = predict_all_items_score_for_user(user_id, self._rating_matrix, self._similarity_matrix, cold=10, k=k)
        predict_result = sorted(pr.items(), key=lambda x: -x[1])
        end = time.time()
        # print(f"Predict Score: ({len(pr.keys())} Items)")
        # print(pr, "\n")
        print("Time Cost of Top N Recommend: %.2f s" % (end - start))
        return predict_result[:n]

    def score_predict_val(self, k):
        """
        验证测试函数
        input:
            k: Integer 预测分数时考虑的相似用户个数
        return:
            res: [] of (User ID, Item ID, Truth Score, Predict Score)
            rmse: Float 度量结果
        """
        if not self._val:
            raise Exception("Only 'val' mode can call this method.")

        res = []
        rmse = 0  # 均方根误差
        count = 0
        start = time.time()
        for i in self._mask_ground_truth.index:
            for j in self._mask_ground_truth.columns:
                if self._mask_ground_truth.loc[i, j] > 0:
                    truth_score = self._mask_ground_truth.loc[i, j]
                    predict_score = predict_item_score_for_user(i, j, self._rating_matrix, self._similarity_matrix, k=k)
                    if predict_score is not None:
                        rmse += (truth_score - predict_score) ** 2
                        count += 1
                        res.append((i, j, truth_score, predict_score))
                        # print(f"[User {i}, Item {j}] TruthScore: {truth_score} PredictScore: {predict_score}")
        rmse = (rmse / count) ** 0.5
        end = time.time()
        print("Time Cost of Score Predict Val: %.2f s " % (end - start))
        return res, rmse
