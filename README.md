# gpu-programming
This repository contains code for a GPU Programming course in NSU
## Лабораторная работа 1
### Задание

Выделить на GPU массив 10^8 элементов типа float и инициализировать их значениями sin((i% 360) * Pi/180). Скопировать результат в память CPU и рассчитать ошибку как sum_i(abs (sin((i% 360) * Pi/180) - arr [i]))/10^8. Проверить значение ошибки для функций sin, sinf, __ sinf, а также для массива элементов типа double.

| | sin  | sinf | __sinf |
| float | 0.500000 | 0.835710 | 8.016642 |
| double | 0.000000 | 0.856817 | 12.213225 | 
