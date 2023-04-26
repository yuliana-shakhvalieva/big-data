## Homework_2

Basic Spark.

#### Standalone Spark:
Standalone cluster Spark был развернут на WSL2 с помощью [данной статьи](https://medium.com/mlearning-ai/single-node-spark-pyspark-cluster-on-windows-subsystem-for-linux-wsl2-22860888a98d). Для поднятия 2 workers в файл ```conf/spark-env.sh``` было добавлено ```SPARK_WORKER_INSTANCES=2```. Подключение к кластеру была выполнено с помощью Jupyter Notebook.

#### Работа с данными на Spark:

Данные были скачаны на компьютер и объединены с помощью метода ```unionByName```. Для тестирования скорости чтения данных и занимаемого объема объединенный датафрейм был сохранен в двух форматах: ```csv, parquet```. Данные в формате ```csv``` заняли почти в два раза больше места, чем те же данные в формате ```parquet``` (```1,1 ГБ``` против ```675 МБ```). Касательно скорости чтения: формат ```csv``` читался быстрее, чем  ```parquet```. Это объясняется их структурой.

В качестве интересного инсайта было выбрано: средний рейтинг книг по годам издания за последние `10` лет. Можно наблюдать, что в основном средний рейтинг книг, выпущенных за год: от ```2.5``` до `3.0`. Тем не менее, в `2021` году средний рейтинг был очень низким (```0.92```), а в ```2022``` достаточно высоким: ```3.89```. Возможно, это связано с качеством данных.

#### Spark Streaming:

Исходные данные были прочитаны с помощью `readStream`. Для расчета среднего рейтинга столбец с отзывами был заменен на численные значения, также был добавлен столбец, отслеживающий время. Обработка данных осуществлялась с помощью функции `foreach_batch_function`. Был выполнен запрос на расчет среднего рейтинга книги. Его результат был записан в файл в формате `parquet`. 