import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F


class My(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(My, self).__init__()
        self.dis_path = dis_path
        # print('路径:',dis_path)
        self.txt_file_name = txt_file_name
        # print('标签:',txt_file_name)
        self.transform = transform
        # print('list_name',list_name)
        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:
            for line in listFile:
                dis, score = line.split()
                dis = dis[:-1]
                # print('dis和score分别',dis,score)
                # print('dis[1:3]',dis[1:3])

                if dis[1:3]  in list_name:

                    score = float(score)
                    # print('score',type(score))
                    dis_files_data.append(dis)
                    # print('dis_files_data',dis_files_data)
                    score_data.append(score)
                    # print('score_data',score_data)

        # reshape score_list (1xn -> nx1)
        # print('score_data1:',score_data)
        score_data = np.array(score_data)
        # print('score_data2:',score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        # print('data是：',data)
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        d_img_path = os.path.join(self.dis_path, d_img_name)

        d_img = cv2.imread(d_img_path, cv2.IMREAD_COLOR)
        if d_img is None:
            raise ValueError(f"无法读取路径为 {d_img_path} 的图像。")

        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
