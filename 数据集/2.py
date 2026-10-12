import re
from cjklib.characterlookup import CharacterLookup


def sort_hanzi_by_stroke(input_path, output_path):
    lookup = CharacterLookup('C')

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 提取汉字
    hanzi_list = re.findall(r'[\u4e00-\u9fff]', text)

    # 汉字和笔画数配对
    hanzi_with_strokes = []
    for char in hanzi_list:
        try:
            strokes = lookup.getStrokeCount(char)
            if strokes is None:
                strokes = 0  # 如果查不到，设为0
        except:
            strokes = 0
        hanzi_with_strokes.append((char, strokes))

    # 按笔画数排序
    hanzi_sorted = sorted(hanzi_with_strokes, key=lambda x: x[1])

    # 提取排序后的汉字
    sorted_chars = [char for char, stroke in hanzi_sorted]

    # 保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(sorted_chars))


# 使用示例
input_file = 'your_input.txt'  # 输入文件路径
output_file = 'sorted_output.txt'  # 输出文件路径

sort_hanzi_by_stroke(input_file, output_file)
