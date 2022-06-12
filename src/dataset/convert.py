from cProfile import label
import scipy.io as scipy
import numpy as np
import cv2
import os

DATA_DIR = "C:/Users/ammar/Documents/CodingProjects/HandGestures/src/dataset"


def get_old_format(mat) -> list:
    boxes = []

    for i, box in enumerate(mat["boxes"][0]):
        # print(f"This is {i=} {box}\n")
        for j, values in enumerate(box[0]):
            # print(f"Onto {j=} {values}")
            try:
                if len(values[4]) == 0:
                    continue
            except:
                pass
            boxes.append(
                [
                    np.int32(values[0][0]),
                    np.int32(values[1][0]),
                    np.int32(values[2][0]),
                    np.int32(values[3][0]),
                ]
            )
    return boxes


def show_labels(img, x_pts, y_pts):
    pts = np.column_stack([x_pts, y_pts])
    print(pts)

    # pts = pts.reshape((-1, 1, 2))
    print(pts)
    img = cv2.polylines(
        img,
        pts=[pts],
        isClosed=True,
        color=(255, 0, 0),
        thickness=2,
    )

    return img


if __name__ == "__main__":
    for folder in os.listdir(DATA_DIR):
        if folder == "convert.py":
            continue
        label_path = f"{DATA_DIR}/{folder}/annotations"

        for old_labels in os.listdir(label_path):
            print(f"{label_path}/{old_labels}")
            mat = scipy.loadmat(f"{label_path}/{old_labels}")
            old_format_boxes = np.array(get_old_format(mat), dtype=object)

            img = cv2.imread(
                f"{DATA_DIR}/{folder}/images/{old_labels.split('.')[0]}.jpg"
            )
            h, w, _ = img.shape

            try:
                os.mkdir(f"{DATA_DIR}/{folder}/labels")
            except:
                pass
            filename = f"{old_labels.split('.')[0]}.txt"
            file = open(f"{DATA_DIR}/{folder}/labels/{filename}", "w")

            for box in old_format_boxes:

                obj_class = 0

                x_pts = np.array([box[0][1], box[1][1], box[2][1], box[3][1]])
                x1, x2 = np.min(x_pts), np.max(x_pts)

                y_pts = np.array([box[0][0], box[1][0], box[2][0], box[3][0]])
                y1, y2 = np.min(y_pts), np.max(y_pts)

                width = float(x2 - x1)
                height = float(y2 - y1)
                X = float(width // 2 + x1) / w
                Y = float(height // 2 + y1) / h

                width /= w
                height /= h

                if X < 0:
                    X = 0
                if Y < 0:
                    Y = 0

                # img = show_labels(img, x_pts, y_pts)

                file.write(f"{obj_class} {X} {Y} {width} {height}\n")
            file.close()

            # cv2.imshow("test", img)
            # cv2.waitKey(0)
