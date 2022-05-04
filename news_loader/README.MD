## Load articles

Скрипт `load_articles.py` загружает статьи с сайта investing.com, связанные с переданным тикером.  
Запуск: `python3 load_articles.py <ticker> <page_begin> <page_end> <save_step>`, где:  
- `ticker` - тикер актива
- `page_begin-page_end` - интервал номеров страниц со списком новостей для скачивания
- `save_step` - шаг с которым сохраняются скачанные новости в файл (необязательный параметр, помогает в случае нестабильного интернета)  

По завершении в директории `data` появится файл вида `<ticker>_<page_begin>_<page_end>.csv` с данными всех статей

Загруженные данные по крупным американским компаниям можно скачать на диске: https://drive.google.com/drive/folders/1jpA_0uMMXvBpPo9lTpMA-Li7FF7MRITE?usp=sharing

