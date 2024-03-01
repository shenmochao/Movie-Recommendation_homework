import pandas as pd
import os
import numpy as np


# ————————————————————————数据预处理————————————————————————————————

def load_data_to_matrix(file_path, step=","):
    """
    读取 载入用户-项 评分表文件，转化为 DataFrame 输出
    input:
        file_path: String 数据文件路径
        step: String 文件分隔符，默认","
    output:
        rating_matrix: DataFrame 评分矩阵
    """
    print(f"Load Data From {file_path} to matrix")
    data = pd.read_csv(file_path, dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32},
                       usecols=range(3), sep=step, engine='python')
    rating_matrix = data.pivot_table(index=["userId"], columns="movieId", values="rating")
    print(f"Shape of Rating Matrix (Users, Movies): {rating_matrix.shape}")
    return rating_matrix


def save_matrix_to_pickle(matrix, dir_path, file_name):
    """
    将 DataFrame 压缩保存至压缩文件中
    input:
        matrix: DataFrame 待保存的矩阵数据
        dir_path: String 文件夹路径
        file_name: String 文件名，不含后缀
    """
    if os.path.exists(dir_path) is False:
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_name + ".pkl")
    print("Save Matrix to", file_path)
    matrix.to_pickle(file_path)


def load_matrix_from_pickle(file_path):
    """
    读取压缩过的 DataFrame 并返回
    input:
        file_path: String 文件路径
    return:
        matrix: DataFrame 矩阵数据
    """
    print("Load Matrix from", file_path)
    matrix = pd.read_pickle(file_path)
    return matrix
