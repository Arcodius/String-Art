import cv2
import numpy as np
import sys
import math

MAX_VALUE = sys.maxsize
SIZE = 128
IMG_SIZE = 256


def diffEdge(start_point, end_point):
    return (start_point[0] - end_point[0]) and (start_point[1] - end_point[1])
def generate_square_pts(size):
    return np.array(
        [[x, 0] for x in range(size)]
        + [[size - 1, y] for y in range(1, size)]
        + [[x, size - 1] for x in range(size - 2, -1, -1)]
        + [[0, y] for y in range(size - 2, 0, -1)]
    )
def calculate_line(pt1, pt2):
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    if(x1 == x2):
        return None, x1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    
def dist_line(m, b, x0, y0):
    result = (m * x0 + b - y0)/math.sqrt(m*m +1)
    return abs(result)

def dist_to_gray(dist):
    x1 = 0.8
    x2 = 1.5
    return max(min(255 * (dist - x1) / (x2 - x1), 255), 0)
    

# 需要得出线段表达式
# 线段近似等宽，忽略前后距离变化
# 降噪
# for all possible edges: draw and try
# 需要resize超采样然后降采样绘制
# 逐像素：目标图片减去模拟部分

# 两种优化方法：同时优化；先趋近图A，后移动点趋近图B

def main():
    # read target file
    target_image = cv2.imread("target.jpg")
    target_image = cv2.resize(target_image, (SIZE, SIZE), cv2.INTER_LANCZOS4)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    alpha = 0.5
    line_color = 0
    iteration = 10

    # 生成正方形边缘的点列
    square_points = generate_square_pts(SIZE)

    image = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    cur_diff = MAX_VALUE

    cur_point = [0, 0]
    # 尝试绘制线，对于所有连线选择一个最小的连线，直到达到某个次数或者cost小于某个阈值
    for iter_time in range(iteration):
        print(iter_time)
        
        min_point = []
        min_diff = MAX_VALUE

        for point in square_points:
            if diffEdge(cur_point, point):
                slope, intercept = calculate_line(cur_point,point)


        k=0
        for start_point in square_points:
            for end_point in square_points:
                if diffEdge(start_point, end_point):
                    # draw new line on duplicated image
                    tmp_img = image.copy()
                    cv2.line(tmp_img, start_point, end_point, color=0, thickness=1)
                    cv2.imshow("Approximated Image", tmp_img)
                    # cv2.waitKey(0)
                    # 全图比较，计算量可以优化
                    # 计算绝对值的差
                    diff_image = cv2.absdiff(tmp_img, target_image)
                    sum_diff = np.sum(diff_image)
                    k+=1
                    
                    # update image
                    if sum_diff < min_diff:
                        min_diff = sum_diff
                        min_points.clear()
                        min_points.append(start_point)
                        min_points.append(end_point)
                        # print(f"{x} {k} sum_diff: {sum_diff}")

        print(f"start_point: {min_points[0]}, end point: {min_points[1]}, current diff: {min_diff}")
        cv2.line(image, min_points[0], min_points[1], color=0, thickness=1)

    # 保存图片
    cv2.imwrite("output.png", image)
    cv2.imshow("Approximated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def greedyCalculate():
    target_image = cv2.imread("target.jpg")
    target_image = cv2.resize(target_image, (SIZE, SIZE), cv2.INTER_LANCZOS4)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    image = np.ones((SIZE, SIZE), dtype=np.uint8) * 255

    alpha = 0.5
    line_color = 0
    iteration = 2000

    for iter_time in range(iteration):
        # blackest two points
        min_val1, _, min_loc1, _ = cv2.minMaxLoc(target_image)
        target_image[min_loc1[1], min_loc1[0]] = 255
        min_val2, _, min_loc2, _ = cv2.minMaxLoc(target_image)
        target_image[min_loc1[1], min_loc1[0]] = min_val1
        
        slope, intercept = calculate_line(min_loc1, min_loc2)
        
        if slope is not None:
            # print(f"直线方程: y = {slope}x + {intercept}")
            for x in range(SIZE):
                y = int(slope * x + intercept)
                if y < 0 or y >= 128: continue

                value = image[y,x] - 10
                if value < 0: 
                    value = 0
                image[y, x] -=10
                if image[y,x]<0:
                    image[y,x] = 0

                t_value = target_image[y,x]
                if t_value > 255: 
                    t_value = 255
                target_image[y,x] +=10
                if target_image[y,x] > 255:
                    target_image[y,x] = 255
        else:
            # print(f"垂直线方程: x = {intercept}")
            for y in range(SIZE):
                image[y, intercept]-=10
                target_image[y,intercept] += 10

    
    #result = cv2.resize(image, (SIZE*2, SIZE*2), cv2.INTER_LANCZOS4)
    cv2.imshow("Approximated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    #main()
    greedyCalculate()