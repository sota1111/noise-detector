import re
import shutil
import os

def comment_out_specific_lines(file_path):
    """
    指定されたファイル内の特定の行に "// " を付けてコメントアウトし、
    元のファイルを上書きする関数
    """
    target_lines = [
        'void *(*rt_variable_malloc_func)(size_t size) = malloc;',
        'void (*rt_variable_free_func)(void *ptr) = free;',
        'void *(*rt_malloc_func)(size_t size) = malloc;',
        'void (*rt_free_func)(void *ptr) = free;'
    ]

    # ファイルの内容を一時的に保存するリスト
    modified_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            # 行の前後の空白を除去して対象行と比較
            if line.strip() in target_lines:
                modified_lines.append("// " + line)
            else:
                modified_lines.append(line)

    # 同じファイルに上書き
    with open(file_path, 'w', encoding='utf-8') as outfile:
        outfile.writelines(modified_lines)


def add_const_to_validation_parameters(file_path):
    """
    ファイル内の 'float Validation_parameterXX[]' 形式に対して、
    const が付いていない場合に const を追加し、同じファイルを上書きする関数
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # "const" が付いていない float Validation_parameterXX[] に対して const を追加
    pattern = r'(?<!const\s)float\s+(Validation_parameter\d+\s*\[\])'
    replacement = r'const float \1'
    modified_content = re.sub(pattern, replacement, content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    print(f"'{file_path}' を上書きしました。")

def move_files_to_validation():
    """
    output_csrc にある4つのファイルを 1つ上の階層の inference/src/Validation に上書きで移動する
    """
    src_dir = "output_csrc"
    dst_dir = os.path.join("..", "inference", "src", "Validation")
    filenames = [
        "Validation_inference.c",
        "Validation_inference.h",
        "Validation_parameters.c",
        "Validation_parameters.h"
    ]

    for filename in filenames:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)  # 上書きでコピー
        print(f"{filename} を {dst_dir} に移動しました（上書き）。")

if __name__ == '__main__':
    file_path = "output_csrc/Validation_inference.c"
    comment_out_specific_lines(file_path)
    file_path = "output_csrc/Validation_parameters.c"
    add_const_to_validation_parameters(file_path)

    move_files_to_validation()
