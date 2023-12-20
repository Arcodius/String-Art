import cv2
import numpy as np
import sys

MAX_VALUE = sys.maxsize
SIZE = 64
IMG_SIZE = 256


def diffEdge(start_point, end_point):
    return (start_point[0] - end_point[0]) and (start_point[1] - end_point[1])

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


    # 生成正方形边缘的点列
    square_points = np.array(
        [[x, 0] for x in range(SIZE)]
        + [[SIZE - 1, y] for y in range(1, SIZE)]
        + [[x, SIZE - 1] for x in range(SIZE - 2, -1, -1)]
        + [[0, y] for y in range(SIZE - 2, 0, -1)]
    )

    cur_image = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    cur_diff = MAX_VALUE

    # 尝试绘制线，对于所有连线选择一个最小的连线，直到达到某个次数或者cost小于某个阈值
    for x in range(50):
        min_points = []
        min_diff=MAX_VALUE
        print(x)
        k=0
        for start_point in square_points:
            for end_point in square_points:
                if diffEdge(start_point, end_point):
                    # draw new line on duplicated image
                    tmp_img = cur_image.copy()
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
        cv2.line(cur_image, min_points[0], min_points[1], color=0, thickness=1)

    # 保存图片
    cv2.imwrite("output.png", cur_image)
    cv2.imshow("Approximated Image", cur_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cur_image = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    # cur_image1 = np.ones((SIZE, SIZE), dtype=np.uint8) * 255
    # diff_image = cv2.absdiff(cur_image, cur_image1)
    # cv2.imshow("Approximated Image", diff_image)
    # print(np.sum(diff_image))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()