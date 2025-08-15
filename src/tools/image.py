import numpy
from PIL import Image

import utils


def load(file):
    img = Image.open(file)
    return numpy.array(img)


def export(data: numpy.ndarray, out_f, fmt='png'):
    img = Image.fromarray(data)
    img.save(out_f, format=fmt)


if utils.module.installed("cv2"):
    import cv2


    def resize(img: numpy.ndarray, size):
        # To shrink an image, it will generally look best with INTER_AREA interpolation,
        # whereas to enlarge an image, it will generally look best with INTER_CUBIC (slow)
        # or INTER_LINEAR (faster but still looks OK).
        img = cv2.resize(img, tuple(reversed(size)), interpolation=cv2.INTER_AREA)
        return img


    def interpolation(img, inter_coord, y, x):
        direct = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        val = 0
        count = 0

        for dy, dx in direct:
            if (y + dy < 0) or (x + dx < 0) or (y + dy >= img.shape[0]) or (x + dx >= img.shape[1]):
                return img[y, x]

            if ((y + dy, x + dx) in inter_coord) or numpy.all(img[y + dy, x + dx] == -1):
                continue

            val += img[y + dy, x + dx]
            count += 1

        if count > 0:
            return val / count
        return img[y, x]


    def get_distortion_map(img_size, K, D):

        p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, tuple(reversed(img_size)), None)
        map1, _ = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), p, tuple(reversed(img_size)), cv2.CV_16SC2)
        reverse_map = -1 * numpy.ones_like(map1)

        # reverse map
        for y in range(map1.shape[0]):
            for x in range(map1.shape[1]):
                index_y, index_x = map1[y, x]
                reverse_map[index_x, index_y, 0] = x
                reverse_map[index_x, index_y, 1] = y

        inter_coord = set()
        for y in range(reverse_map.shape[0]):
            for x in range(reverse_map.shape[1]):
                if numpy.all(reverse_map[y, x] == -1):
                    inter_coord.add((y, x))
                    reverse_map[y, x] = interpolation(reverse_map, inter_coord, y, x)

        return reverse_map


    def distortion(img, reverse_map):
        assert img.shape[:2] == reverse_map.shape[:2]
        distortion_img = cv2.remap(img, reverse_map, numpy.zeros_like(reverse_map[:, :, 0]),
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return distortion_img
