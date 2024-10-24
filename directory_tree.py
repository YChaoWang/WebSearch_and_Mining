import os


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):

        # 排除 .git 目錄
        dirs[:] = [d for d in dirs if d != ".git"]
        level = root.replace(startpath, "").count(os.sep)
        indent = "├── " if level > 0 else ""
        print(f"{'│   ' * (level - 1)}{indent}{os.path.basename(root)}/")
        subindent = "│   " * level + "├── "
        for i, f in enumerate(files):
            if f.endswith(".py") or f == "requirements.txt" or f == "english.stop":
                file_indent = (
                    subindent if i < len(files) - 1 else "│   " * level + "└── "
                )
                print(f"{file_indent}{f}")


if __name__ == "__main__":
    path = "."  # 替換為你的目錄路徑
    list_files(path)
