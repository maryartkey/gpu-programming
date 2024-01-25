## Лабораторная работа 2
### Задание
Implement a program for applying filters to your images. Possible filters: blur, edge detection, denoising. Implement three versions of the program, namely, using global, shared memory and texture. Compare the time. To work with image files, it is recommended to use libpng (man libpng).

### Результаты
<img src = "https://github.com/maryartkey/gpu-programming/assets/35896507/9145a78b-d419-4ad1-91bf-f80d15a427b5" width = 300>
<img src = "https://github.com/maryartkey/gpu-programming/assets/35896507/5d724dff-01b2-4570-b2da-6fe3c0dca9d9" width = 300>

| Память | Time, мкс |
|:--------:|------:|
|  global  | 3421 |
|  shared  | 3304 |
|  texture | 7106 |
