## Лабораторная работа 4
### Задание
Use the method of least squares to find a circle in the image. For each random sample points organize their processing on the GPU. Random sampling arrange with library CURAND.

Input: an image size of (for example, a speed limit sign), the number of samples, the number of elements in each sample.
Output: The image with a circle painted on it.

Recommendation: before processing apply Sobel filter for edge detection and consider the point at which the normalized color value>0.5.

### Решение
### Фильтр Собеля
Фильтр Собеля использует два ядра 3x3. Одно ядро предназначено для горизонтальных изменений, а другое — для вертикальных. Когда эти ядра применяются к изображению, они создают два производных изображения, которые можно объединить, чтобы найти абсолютную величину градиента в каждой точке. Фильтр Собеля выделяет края изображения, которые представляют собой области с высоким градиентом.

### Метод наименьших квадратов

Основная идея использования метода наименьших квадратов для поиска кругов на изображении состоит в том, чтобы привести уравнение окружности (x-a)^2 + (y-b)^2 = r^2 к набору точек, где (a,b) — центр круга, а r — радиус. Поиск этих точек включает решение системы уравнений, полученной из уравнения круга.

### Алгоритм

1) Загружаем изображение с кругом.
2) Применяем фильтр Собеля для поиска границ. Выполняем параллельно на ГП.
3) Выбираем все точки, которые вероятно относятся к выделенным границам (подбираем normalized color value).
4) Ищем на изображении круги методом наименьших квадратов: берем N раз по K случайных сгенерированных точек, пытаемся строить окружность и оценивать ее, формируем short list и выбираем лучшего из кандидата.

<img src = "https://github.com/maryartkey/gpu-programming/assets/35896507/dfe752c4-1397-4bc0-9302-e33f983ac8f3" width = 300>
