import os
import csv
import argparse
import cv2
import pandas as pd
import imagehash
import time
from PIL import Image
from query_image import judge_pic


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


def cut_video_for_once(file_path, save_path):
    path_dir = os.listdir(file_path)  # 获取文件夹中文件名称
    if not os.path.exists(save_path):  # 如果不存在就创建文件夹
        print('创建文件夹', save_path)
        os.makedirs(save_path)  # 防止不存在保存文件路径，先创建该文件夹
    del_files(save_path)
    i = 1
    num = len(path_dir)
    for all_dir in path_dir:  # 逐个读取视频文件
        video_path = os.path.join(file_path, all_dir)
        vc = cv2.VideoCapture(video_path)  # 读入视频文件
        vc.set(cv2.CAP_PROP_POS_FRAMES, 50)  # 设置要获取的帧号
        TorF, frame = vc.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        print('视频', i, '预处理完成，共', num, '个视频')
        i += 1
        per_name = str(all_dir[:-4]) + '.jpg'
        cv2.imwrite(os.path.join(save_path, per_name), frame)
    return 0


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
    add_jpg = [str(t) + '.mp4' for t in dir_name_list]  # 添加jpg后缀
    add_video = ['video' + str(t) + '.mp4' for t in target_list]  # 添加video前缀与MP4后缀
    degree_of_confidence = [round((100 - t) / 101, 4) for t in all_dist]  # 计算置信度

    # 写入csv文件
    print('计时结束')
    test = pd.DataFrame({'指定视频': add_jpg, '检出视频': add_video, '置信度': degree_of_confidence})
    print('已写入', os.path.join(result_path, 'result.csv'), '文件，编码集为gbk')
    test.to_csv(os.path.join(result_path, 'result.csv'), encoding='gbk', index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, help='请输入文件夹名称')
    parser.add_argument('--result', type=str, help='请输入文件夹名称')
    parser.add_argument('--video', type=str, help='请输入文件夹名称')
    args = parser.parse_args()

    hash_data_path = os.path.join(args.cache, 'hash_data', 'data.csv')
    video_res_path = os.path.join(args.result, 'video_res_path')
    video_cut_path = os.path.join(args.cache, 'video_cut_path')
    video_pre_path = os.path.join(args.cache, 'video_pre_path')

    print('计时开始，正在预处理请稍后')
    time_start = time.time()  # 计时开始
    cut_video_for_once(args.video, video_cut_path)
    print('加载预处理文件,请稍后')
    judge_pic(video_cut_path, video_pre_path)
    auto_detection(video_pre_path, video_res_path, hash_data_path)
    time_end = time.time()  # 计时结束

    time_cost = time_end - time_start
    with open(os.path.join(video_res_path, 'time_cost.txt'), 'w', encoding='gbk') as f:
        f.write('所在文件夹所有图片匹配总耗时' + str(round(time_cost * 1000, 2)) + '毫秒')
    print('已写入', os.path.join(video_res_path, 'time_cost.txt'), '文件，编码集为gbk')
    print('匹配完毕，请前往', video_res_path, '文件夹查看')
    return


if __name__ == '__main__':
    main()
