import json

# 读取jsonl文件,逐行解析JSON数据
translations = []
with open("./i18n.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        translations.append(json.loads(line))

# 获取所有的语言代码
languages = list(translations[0].keys())

# 为每个语言创建一个字典
lang_dicts = {lang: {} for lang in languages}

# 遍历每个翻译条目
for item in translations:
    # 遍历每个语言
    for lang in languages:
        # 获取当前语言的单词
        word = item.get(lang)
        if word:
            # 遍历其他语言
            for other_lang in languages:
                if other_lang != lang:
                    # 获取其他语言的单词
                    other_word = item.get(other_lang)
                    if other_word:
                        # 将其他语言的单词添加到当前语言单词的数组中
                        lang_dicts[lang].setdefault(word, []).append(other_word)

# 为每个语言生成JSON文件
for lang, lang_dict in lang_dicts.items():
    # 生成JSON文件名
    filename = f"{lang}.json"

    # 将字典保存为JSON文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(lang_dict, f, ensure_ascii=False, indent=4)

    print(f"生成 {filename} 成功")