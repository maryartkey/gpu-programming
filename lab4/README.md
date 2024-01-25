## Лабораторная работа 4
### Задание
Use the method of least squares to find a circle in the image. For each random sample points organize their processing on the GPU. Random sampling arrange with library CURAND.

Input: an image size of (for example, a speed limit sign), the number of samples, the number of elements in each sample.
Output: The image with a circle painted on it.

Recommendation: before processing apply Sobel filter for edge detection and consider the point at which the normalized color value>0.5.
