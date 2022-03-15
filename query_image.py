import csv
import imagehash
import os
import pandas as pd
import time
import cv2
from PIL import Image
import argparse


def del_files(path_file):
    if os.path.exists(path_file):
        ls = os.listdir(path_file)
        for i in ls:
            f_path = os.path.join(path_file, i)
            # 判断是否是一个目录,若是,则递归删除
            if os.path.isdir(f_path):
                del_files(f_path)
            else:
                os.remove(f_path)


def auto_core(img_dir_name, rows):
    dirs = os.listdir(img_dir_name)
    target_list = []  # 存储所有匹配图片
    dir_name_list = []  # 存储所有图片文件名
    all_dist = []  # 存储所有海明距
    t = 0  # 记录循环数量的变量

    for dir in dirs:
        min_dict = {}  # 存储3185个最小海明距离的字典
        test_hash = imagehash.whash(Image.open(dir).convert('L'), hash_size=16)  # 测试图片的whash值保存为test_hash
        for i in range(len(rows)):
            name = rows[i][0]  # 存储每行第一个文件名
            pic_dist = []
            for j in range(1, len(rows[i])):
                if rows[i][j] == '':  # 存在不足25个的情况跳过
                    continue
                if '.' not in rows[i][j]:  # 存在数据溢出的情况下跳过
                    img_hash = imagehash.hex_to_hash(rows[i][j])  # 将csv文件中读出的指定whash值转换为imagehash数据类型
                    if img_hash.hash.shape != test_hash.hash.shape:  # 存在维数不同的情况下跳过
                        continue
                    pic_dist.append(img_hash - test_hash)  # 计算海明距
            min_d = min(pic_dist)
            num = pic_dist.count(min_d)

            min_dict[name] = (min_d, num)  # 保存该n号文件夹中与指定图片海明距最小的距离，i号键对应i号文件夹的最小海明距

        target = min(min_dict, key=lambda k: min_dict[k][0])
        min_di = min_dict[target][0]
        max_num = 0
        for k, v in min_dict.items():
            if v[0] == min_di:
                if v[1] > max_num:
                    max_num = v[1]
                    target = k
        # target： 查找到的拥有min_di个数最多的视频
        # max_num : 有几个
        #  min_di   最小值
        target_list.append(target)
        dir_name_list.append(dir[:-4])
        all_dist.append(min_dict.get(target)[0])
        t += 1
        print('已匹配完成第', t, '张图片，请稍后')

    return target_list, dir_name_list, all_dist


# 指定文件夹自动匹配视频并生成结果文件
def auto_detection(full_path, result_path, csv_file):
    os.chdir(full_path)

    # 验证路径合法
    if not os.path.exists(result_path):  # 如果不存在就创建文件夹,用于存储result.csv 与time_cost。txt
        print('创建文件夹', result_path)
        os.makedirs(result_path)  # 防止不存在保存文件路径，先创建该文件夹

    # 提取csv文件
    os.chdir(full_path)  # 转到img目录
    csvfile = open(csv_file)  # 打开csv文件
    reader = csv.reader(csvfile)
    rows = [row for row in reader]  # 按行读取并全部存储在rows变量中

    # 核心代码

    target_list, dir_name_list, all_dist = auto_core(full_path, rows)  # 运行核心查询代码

    # 更改待写入变量
    add_jpg = [str(t) + '.jpg' for t in dir_name_list]  # 添加jpg后缀
    add_video = ['video' + str(t) + '.mp4' for t in target_list]  # 添加video前缀与MP4后缀
    degree_of_confidence = [round((100 - t) / 101, 4) for t in all_dist]  # 计算置信度

    # 写入csv文件
    print('计时结束')
    test = pd.DataFrame({'指定图片': add_jpg, '检出视频': add_video, '置信度': degree_of_confidence})
    print('已写入', os.path.join(result_path, 'result.csv'), '文件，编码集为gbk')
    test.to_csv(os.path.join(result_path, 'result.csv'), encoding='gbk', index=False)


# 过滤crop与bp图片
def deal_bp_and_shape(full_pic_path, write_path):
    img = cv2.imread(full_pic_path, cv2.IMREAD_GRAYSCALE)  # 读取图片
    # img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    x = binary_image.shape[0]  # 高度
    y = binary_image.shape[1]  # 宽度
    bottom = i = top = 0
    # 自上而下自左而右
    while i < x:
        for j in range(y):
            if binary_image[i][j] != 0:
                top = i - 1
                i = x
                break
        i += 1
    i = 0
    while i < x:
        for j in range(y):
            if binary_image[x - i - 1][j] != 0:
                bottom = x - i - 1
                i = x
                break
        i += 1

    height = bottom - top + 1  # +1防止纯黑图，高度为0造成无法读写
    # x-height:要切除的高度
    if 100 < x - height:
        # print('图片判断为黑边图片应该切除，切除后高度为', height, '原高度为', x)
        pre1_picture = img[top:top + height, 0:y]  # 图片截取
        cv2.imwrite(write_path, pre1_picture)

    else:
        # print('判断为crop图片将执行填充，高度为', height, '原高度为', x)
        top_size, bottom_size, left_size, right_size = (
            (720 - x) // 2, (720 - x) // 2, (1280 - y) // 2, (1280 - y) // 2)
        cv2.imwrite(write_path, cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,
                                                   borderType=cv2.BORDER_REPLICATE))


def judge_pic(image_path, img_pre_path):
    os.chdir(image_path)
    files = os.listdir(image_path)
    if not os.path.exists(img_pre_path):  # 如果不存在就创建文件夹,用于存储result.csv 与time_cost。txt
        print('创建文件夹', img_pre_path)
        os.makedirs(img_pre_path)  # 防止不存在保存文件路径，先创建该文件夹
    del_files(img_pre_path)
    for file in files:
        write_path = os.path.join(img_pre_path, file)
        img = cv2.imread(os.path.join(image_path, file))
        if 1.599 < img.shape[1] / img.shape[0] < 1.801:
            # print(file, '属于裁切较小crop不处理')
            cv2.imwrite(write_path, img)
        elif img.shape[1] < img.shape[0]:
            # print(file, '属于长视频不处理')
            cv2.imwrite(write_path, img)
        else:
            # print(file, '属于特殊情况，处理并覆盖')
            deal_bp_and_shape(os.path.join(image_path, file), write_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, help='请输入文件夹名称')
    parser.add_argument('--result', type=str, help='请输入文件夹名称')
    parser.add_argument('--img', type=str, help='请输入文件夹名称')
    args = parser.parse_args()

    hash_data_path = os.path.join(args.cache, 'hash_data', 'data.csv')
    img_pre_path = os.path.join(args.cache, 'img_pre_path')
    img_res_path = os.path.join(args.result, 'img_res_path')

    time_start = time.time()  # 计时开始
    print('计时开始，正在预处理请稍后')
    judge_pic(args.img, img_pre_path)
    auto_detection(img_pre_path, img_res_path, hash_data_path)
    time_end = time.time()  # 计时结束

    # 写入txt文件
    time_cost = time_end - time_start
    with open(os.path.join(img_res_path, 'time_cost.txt'), 'w', encoding='gbk') as f:
        f.write('所在文件夹所有图片匹配总耗时' + str(round(time_cost * 1000, 2)) + '毫秒')
    print('已写入', os.path.join(img_res_path, 'time_cost.txt'), '文件，编码集为gbk')
    print('匹配完毕，请前往', img_res_path, '文件夹查看')
    return


if __name__ == '__main__':
    main()
