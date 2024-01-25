## Лабораторная работа 2
### Задание
Implement a program for applying filters to your images. Possible filters: blur, edge detection, denoising. Implement three versions of the program, namely, using global, shared memory and texture. Compare the time. To work with image files, it is recommended to use libpng (man libpng).

### Решение
#### Глобальная память
Версия с глобальной памятью предполагает, что каждый поток загружает данные изображения непосредственно из глобальной памяти, обрабатывает их и записывает результат обратно в глобальную память.
#### Разделяемая память
В версии с общей памятью каждый блок потоков загружает часть изображения в общую память. Это снижает использование полосы пропускания глобальной памяти.
#### Текстурная память
Память текстур обеспечивает кэширование, что полезно для операций только чтения с пространственной локальностью. Текстуры CUDA считываются с помощью функций выборки текстур. Сначала входное изображение привязывается к текстуре CUDA, затем, аналогично версии глобальной памяти, оно считывается из текстуры при помощи tex2D(tex, x, y).

### Результаты
<img src = "https://github.com/maryartkey/gpu-programming/assets/35896507/9145a78b-d419-4ad1-91bf-f80d15a427b5" width = 300>
<img src = "https://github.com/maryartkey/gpu-programming/assets/35896507/5d724dff-01b2-4570-b2da-6fe3c0dca9d9" width = 300>

| Память | Time, мкс |
|:--------:|------:|
|  global  | 3421 |
|  shared  | 3304 |
|  texture | 7106 |
