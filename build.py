import cv2 as cv
import os
import imagehash
import pandas as pd
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
        ls = os.listdir(path_file)
        for i in ls:
            os.rmdir(os.path.join(path_file, i))


def cut_video(file_path, save_path):
    path_dir = os.listdir(file_path)  # 获取文件夹中文件名称
    if not os.path.exists(save_path):  # 如果不存在就创建文件夹
        print('创建文件夹', save_path)
        os.makedirs(save_path)  # 防止不存在保存文件路径，先创建该文件夹
    del_files(save_path)
    video_num = len(path_dir)
    ts = 1
    for all_dir in path_dir:  # 逐个读取视频文件
        a = 1  # 图片计数-不改
        c = 1  # 帧数计数-不改
        video_path = os.path.join(file_path, all_dir)
        print('正在预处理第', ts, '个视频:', all_dir, '，共', video_num, '个视频')
        ts += 1
        vc = cv.VideoCapture(video_path)  # 读入视频文件
        # 存储视频的子目录
        path = os.path.join(save_path, all_dir[:-4])
        # print(path)
        if not os.path.exists(path):  # 如果不存在就创建文件夹
            # print('创建文件夹', path)
            os.mkdir(path)
        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False
            print('error,wrong path')

        timeF = 12  # 帧数间隔，控制每n帧截一张图
        while rval:
            rval, frame = vc.read()  # 分帧读取视频
            if rval == False:
                break
            if (c % timeF == 0):
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 保存为灰度图格式
                cv.imwrite(os.path.join(path, str(a) + '.jpg'), frame)

                a = a + 1
            c = c + 1
            cv.waitKey(1)
        vc.release()


def get_hash(main_path, save_path):
    all_hash = []
    if not os.path.exists(save_path):  # 如果不存在就创建文件夹
        print('创建文件夹', save_path)
        os.makedirs(save_path)  # 防止不存在保存文件路径，先创建该文件夹

    files = os.listdir(main_path)  # 读入总文件夹所有子文件夹名称
    for file in files:  # 遍历0，1，2，3号子文件夹
        per_path = os.path.join(main_path, file)
        pic_names = os.listdir(per_path)  # 表示n号文件夹下的每张图片名的列表集合
        pic_names.sort(key=lambda x: int(x[:-4]))  # 图片名进行排序，否则处理顺序变为0，1，11，2，22，
        per_hash = []  # 存储n号文件夹的25张图片的whash值
        per_hash.append(file)
        for pic in pic_names:  # 遍历n号子文件夹中的1，2，3，4号图片
            pic_hash = imagehash.whash(Image.open(os.path.join(per_path, pic)), hash_size=16)  # 得到每张图片的whash值
            per_hash.append(pic_hash)  # 追加在per_hash的末尾，构成n号文件夹的whash值集合
        all_hash.append(per_hash)  # 追加在all_hash的末尾，构成3000个文件夹集合中的whash值集合
        print('第', file, '号视频处理完毕')
    # 存储为data.csv
    df = pd.DataFrame(all_hash)
    os.chdir(save_path)
    df.to_csv('data.csv', index=False, header=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=str, help='请输入文件夹名称')
    parser.add_argument('--cache', type=str, help='请输入文件夹名称')
    args = parser.parse_args()

    video_cut_path = os.path.join(args.cache, 'video_cut')
    hash_data_path = os.path.join(args.cache, 'hash_data')

    cut_video(args.db, video_cut_path)
    get_hash(video_cut_path, hash_data_path)

    print('处理完毕，请运行query_image.py与query_video.py')
    return


if __name__ == '__main__':
    main()
