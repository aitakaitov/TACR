import os
import random
import shutil


domains = os.listdir('../../ad_pages')

for domain in domains:
    os.makedirs(f'{domain}')

    files = os.listdir(f'../../ad_pages/{domain}/p_only_sanitized')
    random.shuffle(files)

    selected_files = files[:5]

    for file in selected_files:
        shutil.copyfile(f'../../ad_pages/{domain}/p_only_sanitized/{file}', f'./{domain}/{file}')

