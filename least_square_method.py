import cv2
import numpy as np
import sys
import math

from bresenham import *
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.optimize import linprog
import cvxpy as cp

MAX_VALUE = sys.maxsize

def build_image_vector(image, size):
    row_ind = []
    col_ind = []
    data = []
    for y, line in enumerate(image):
        for x, pixel_value in enumerate(line):
            pixel_code = y * size + x
            data.append(float(pixel_value))
            row_ind.append(pixel_code)
            col_ind.append(0)
    sparse_b = csr_matrix((data, (row_ind, col_ind)), shape=(size * size, 1))
    return sparse_b

def square_points(size):
    return np.array([[0, y] for y in range(size)] + 
                    [[size-1, y] for y in range(size)] + 
                    [[x, 0] for x in range(1, size-1)] + 
                    [[x, size-1] for x in range(1, size-1)])

def build_adjacency_matrix(size):
    edge_points = square_points(size)
    edge_points = edge_points.astype(int)

    edge_codes = []
    # store all 1D data of lines
    row_ind = []
    col_ind = []
    for i, ni in enumerate(edge_points):
        for j, nj in enumerate(edge_points[i+1:], start=i+1):
            # print(f"i = {i}, ni = {ni}, j = {j}, nj = {nj}")
            # 1D line of i-j id
            edge_codes.append((i, j))
            pixels = bresenham(ni, nj).path
            edge_data = []
            for pixel in pixels:
                pixel_code = pixel[1] * size + pixel[0]
                edge_data.append(pixel_code)
            
            row_ind += edge_data
            col_ind += [len(edge_codes)-1] * len(edge_data)
            # tmp = [len(edge_codes)-1] * len(edge_data)
            # print(f"row_id = {edge_data}, col_id = {tmp}")

    # 创建稀疏矩阵
    sparse = csr_matrix(([255]*len(row_ind), (row_ind, col_ind)), shape=(size*size, len(edge_codes)))
    
    return sparse, edge_points, edge_codes

def rebuild_and_save(x, sparse, size, filename):
    brightness_correction = 1.2
    x *= brightness_correction
    b_approx = sparse.dot(x)
    b_image = b_approx.reshape((size, size))
    b_image = np.clip(b_image, 0, 255)
    
    cv2.imwrite(filename, b_image)

def main():
    SIZE = 256
    
    target_image = cv2.imread("woman.jpg")
    target_image = cv2.resize(target_image, (SIZE, SIZE), cv2.INTER_LANCZOS4)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    target_image = cv2.bitwise_not(target_image)
    
    print(f"building image vector")
    sparse_b = build_image_vector(target_image, SIZE)
    save_npz('sparse_b.npz', sparse_b)

    print(f"building adjacency matrix")
    sparse, edge_points, edge_codes = build_adjacency_matrix(SIZE)
    save_npz('sparse_A.npz', sparse)

    # 求解整数规划问题
    # result = linprog(c, A_eq=sparse, b_eq=sparse_b, bounds=bounds, method='highs')

    print(f"solving...")
    result = lsqr(sparse, np.array(sparse_b.todense()).flatten())
    x = result[0]

    print(f"saving negative jpg")
    rebuild_and_save(x, sparse, SIZE, "negative.jpg")
    x = np.clip(x, 0, 1e6)
    print(f"saving unquantized jpg")
    rebuild_and_save(x, sparse, SIZE, "unquantized.jpg")

    quantization_level = 30 # 50 is already quite good. None means no quantization.
    # clip values larger than clip_factor times maximum.
    # (The long tail does not add too much to percieved quality.)
    clip_factor = 0.3
    if quantization_level is not None:
        max_edge_weight_orig = np.max(x)
        x_quantized = (x / np.max(x) * quantization_level).round()
        x_quantized = np.clip(x_quantized, 0, int(np.max(x_quantized) * clip_factor))
        # scale it back:
        x = x_quantized / quantization_level * max_edge_weight_orig
    print(f"saving quantized jpg")
    rebuild_and_save(x, sparse, SIZE, "quantized.jpg")


if __name__ == "__main__":
    main()