import scipy.io as scipy
import numpy as np
import cv2


# print(list(mat))
# print(mat["boxes"])


def get_old_format(mat) -> list:
    boxes = []

    for i, box in enumerate(mat["boxes"][0]):
        # print(f"This is {i=} {box}\n")
        for j, values in enumerate(box[0]):
            # print(f"Onto {j=} {values}")
            boxes.append(
                [
                    np.int32(values[0][0]),
                    np.int32(values[2][0]),
                    np.int32(values[2][0]),
                    np.int32(values[3][0]),
                    np.int32(values[1][0]),
                    values[4][0],
                ]
            )
    return boxes


if __name__ == "__main__":
    mat = scipy.loadmat("test_dataset/test_data/annotations/VOC2007_1.mat")
    old_format = np.array(get_old_format(mat), dtype=object)
    print(f"{old_format=}")

    img = cv2.imread("test_dataset/test_data/images/VOC2007_1.jpg")
    print("trying this ", (old_format[0][:4]))
    img = cv2.polylines(
        img, pts=old_format[0][:4], isClosed=True, color=255, thickness=2
    )

    cv2.imshow("test", img)
    cv2.waitKey(0)
