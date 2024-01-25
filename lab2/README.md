## Лабораторная работа 2
### Задание
Implement a program for applying filters to your images. Possible filters: blur, edge detection, denoising. Implement three versions of the program, namely, using global, shared memory and texture. Compare the time. To work with image files, it is recommended to use libpng (man libpng).

### Результаты
![cat]("lab2/cat.png")
![cat]("lab2/cat_blurred.png")
| Память | Time, мкс |
|:--------:|------:|
|  global  | 3304 |
|  shared  | 3421 |
|  texture | 9106 |

### Выводы


