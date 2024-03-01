from matplotlib import pyplot as plt
from Collaborative_filtering import *


if __name__ == "__main__":
    # 选择使用的数据集
    data_name = "ml-latest-small"
    # data_name = "ml-1m"

    if data_name == "ml-latest-small":
        cf_i = CF(0.99, data_name, "./Datasets/ml-latest-small/ratings.csv")
    else:
        cf_i = CF(0.8, data_name, "./Datasets/ml-1m/ratings.dat", step="::")

    # 推荐测试
    print("5 Item-based Recommend Movies for User 3:", cf_i.top_n_recommend(3, n=5, k=10))

    # 验证测试可视化
    k_l_i = []
    r = range(1, 100, 1)
    for k in r:
        print(f"Ite: {k}")
        _, rmse_i = cf_i.score_predict_val(k)
        k_l_i.append(rmse_i)

    min_i = np.argmin(k_l_i)
    i, ii = min_i + 1, k_l_i[min_i]

    save_dir = "./Image/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.title("Dataset: " + data_name)
    plt.plot(r, k_l_i, label="Item CF")
    plt.scatter(i, ii)
    plt.annotate("(%d, %.2f)" % (i, ii), xy=(i, ii), xytext=(-20, 10), textcoords='offset points')
    plt.legend()
    plt.savefig(save_dir + data_name + ".png")
    plt.show()
