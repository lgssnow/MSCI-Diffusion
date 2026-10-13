import re


def remove_duplicate_hanzi(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 使用正则表达式提取所有汉字
    hanzi_list = re.findall(r'[\u4e00-\u9fff]', text)

    # 去除重复汉字，保留唯一的汉字
    unique_hanzi = ''.join(sorted(set(hanzi_list), key=hanzi_list.index))

    # 将结果写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(unique_hanzi)


# 用法示例
input_file = '11.txt'  # 你的原文件路径
output_file = '11.txt'  # 结果保存的文件路径

remove_duplicate_hanzi(input_file, output_file)
