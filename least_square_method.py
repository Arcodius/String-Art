import cv2
import numpy as np
import sys
import math
import itertools
import random

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

def square_points(image_size, num):
    # expected number of partitions on each edge and image size
    delta = image_size / num # float value
    a, b, c, d = [], [], [], []
    
    for x in itertools.count(start=0, step=delta):
        if x >= image_size:
            break
        a.append((0, round(x)))
        b.append((image_size-1, round(x)))
        c.append((round(x), 0))
        d.append((round(x), image_size-1))
        # a.append([0, round(x)])
        # b.append([image_size-1, round(x)])
        # c.append([round(x), 0])
        # d.append([round(x), image_size-1])
        
    return a+b+c+d

def diffEdge(start_point, end_point):
    return (start_point[0] - end_point[0]) and (start_point[1] - end_point[1])

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

def compare(pixels, cur_image, target_image):
    value = 0
    diff = 0
    for coords in pixels:
        diff += target_image[coords[0], coords[1]] - cur_image[coords[0], coords[1]]

    return diff

def draw_line(coords, image):
    value = 0
    for coord in coords:
        image[coord[0], coord[1]] = value
    return image

def greedyMethod():
    SIZE = 128
    hooks = square_points(SIZE, 8)

    #存储点对之间的哈希表
    line_hash_table = {}
    for start in hooks:
        for end in hooks:
            line_hash_table[(start, end)] = False

    target_image = cv2.imread("woman.jpg")
    target_image = cv2.resize(target_image, (SIZE, SIZE), cv2.INTER_LANCZOS4)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    image = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    
    #从任意一点出发，进行(n-1)次连线，记录最接近的值

    last_min_diff = MAX_VALUE
    min_diff = MAX_VALUE
    min_pair = []
    begin = random.choice(hooks)
    
    for end in hooks:
        if diffEdge(begin, end):
            hook_pair = (begin, end)
            if hook_pair in line_hash_table:
                if line_hash_table[hook_pair] == False:
                    line_hash_table[hook_pair] = True
                    hook_pair_inv = (end, begin)
                    line_hash_table[hook_pair_inv] = True

                    # need to add grayvalue information
                    pixel_coods = bresenham(begin, end).path

                    cur_image = np.copy(image)
                    cur_image = draw_line(pixel_coods, cur_image)
                    cv2.imshow("Approximated Image", cur_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    diff = compare(pixel_coods, cur_image, target_image)
                    print(diff)
                    if diff < min_diff:
                        min_diff = diff
                        min_pair.clear()
                        min_pair.append(hook_pair)
                    # compare
                    begin = end

            else: 
                print(f"key {hook_pair} not found")

if __name__ == "__main__":
    greedyMethod()