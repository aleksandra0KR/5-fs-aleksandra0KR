# Лабораторная №5. Выбор признаков

## Набор данных
- Выберите набор данных для задачи классификации текста.
- Желательно использовать набор данных, который вы получили в первой лабораторной работе, если он содержал полноценный текстовый признак.
- Если не удастся найти подходящий набор данных, можно взять набор данных  SMS или castle-or-lock.
- Выберите целевую функцию ошибки или качества для задачи классификации, а также соответствующий способ валидации алгоритма классификации.
## Алгоритмы
- Реализуйте 3 метода выбора признаков: встроенный, обёртку и фильтрующий.
- Выберите библиотечную реализацию 3-х методов выбора признаков: встроенного, обёртки и фильтрующего. При этом конкретные варианты методов должны отличаться от ваших реализаций.
## Задание
- Векторизуйте набор данных при помощи CountVectorizer или аналогов.
- Выведите 30 наиболее значимых признаков (слов) различными методами выбора признаков. Сравните полученные списки.
- Определите, как меняется качество работы различных (не менее трёх) классификаторов до и после выбора признаков каждым из методов. Выберите один метод выбора признаков.
- Кластеризуйте любым алгоритмом кластеризации данные до и после выбора признаков. Оцените качество кластеризации любой внешней и внутренней мерой.
- Методами PCA и tSNE уменьшите размерность данных до и после выбора признаков. Визуализируйте данные и отметьте реальные классы, а также как их кластеризовал алгоритм кластеризации из предыдущего пункта.


# Laboratory 5. Selection of features

## Data set
- Select the data set for the text classification task.
- It is advisable to use the data set that you received in the first laboratory work if it contained a full-fledged text feature.
- If you cannot find a suitable data set, you can use the SMS or lock-or-lock data set.
- Select the target error or quality function for the classification task, as well as the appropriate method for validating the classification algorithm.
## Algorithms
- Implement 3 feature selection methods: embedded, wrapper, and filtering.
- Select a library implementation of 3 feature selection methods: embedded, wrapper, and filtering. However, the specific method options should differ from your implementations.
## Assignment
- Vectorize the data set using CountVectorizer or analogs.
- Output the 30 most significant features (words) using various feature selection methods. Compare the received lists.
- Determine how the quality of work of various (at least three) classifiers changes before and after the selection of features by each of the methods. Select one feature selection method.
- Use any clustering algorithm to cluster data before and after feature selection. Evaluate the clustering quality by any external and internal measure.
- Using PCA and tSNE methods, reduce the data dimension before and after feature selection. Visualize the data and mark the real classes, as well as how they were clustered by the clustering algorithm from the previous paragraph.
