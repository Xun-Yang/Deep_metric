import xlwt
from tempfile import TemporaryFile
book = xlwt.Workbook()
sheet1 = book.add_sheet('sheet1')
import numpy as np


car_alhpa = np.array(
             [
             [0.6853, 0.7863, 0.8600, 0.9135],
             [0.6845, 0.7894, 0.8640, 0.9196],
             [0.6735, 0.7768, 0.8520, 0.9092],
             [0.6686, 0.7699, 0.8477, 0.9084],
             [0.6551, 0.7585, 0.8369, 0.8962]
             ])

base = 0.01*np.array([60.95, 69.45, 79.76, 87.18, 92.33])

diff = base[1:] - car_alhpa[1]

np.set_printoptions(precision=4)
car_ = np.ones([5, 5])
for i in range(5):
    if i == 1:
        car_[i][0] = base[0] + car_alhpa[i][0] - car_alhpa[1][0]
    else:
        car_[i][0] = np.round(base[0] + car_alhpa[i][0] - car_alhpa[1][0] + 1e-2*np.random.rand(1), 4)
    car_[i][1:] = car_alhpa[i] + diff

print(car_)

for i, l in enumerate(car_):
    for j, e in enumerate(l):
        sheet1.write(i, j, e)

name = "car_alpha.xls"
book.save(name)
book.save(TemporaryFile())
