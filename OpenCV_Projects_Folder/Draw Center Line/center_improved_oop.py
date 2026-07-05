import cv2


# -----------------------------
# Configuration
# -----------------------------
RESIZE_WIDTH = 800
RESIZE_HEIGHT = 600

CENTER_MARGIN_RATIO = 0.10
CENTER_PADDING = 0.25

BOX_COLOR = (0, 0, 255)
CENTER_LINE_COLOR = (255, 0, 0)

BOX_THICKNESS = 3
LINE_THICKNESS = 2


class ImageProcessor:
    def __init__(self, img_path):
        self.img_path = img_path
        self.img = None
        self.height = None
        self.width = None
        self.channels = None
        self.center_x = None
        self.center_y = None

    def load_img(self):
        self.img = cv2.imread(self.img_path)

        if self.img is None:
            raise FileNotFoundError(f"Image could not be loaded: {self.img_path}")

    def resize_img(self):
        self.img = cv2.resize(self.img, (RESIZE_WIDTH, RESIZE_HEIGHT))

    def get_img_shape(self):
        self.height, self.width, self.channels = self.img.shape

    def calculate_center(self):
        self.center_x = self.width // 2
        self.center_y = self.height // 2

    def draw_center_region(self):
        center_margin = int(self.width * CENTER_MARGIN_RATIO)

        center_padding = max(0.0, min(CENTER_PADDING, 0.49))

        left_x = self.center_x - center_margin
        right_x = self.center_x + center_margin

        top_y = int(self.height * center_padding)
        bottom_y = int(self.height * (1 - center_padding))

        cv2.rectangle(
            self.img,
            (left_x, top_y),
            (right_x, bottom_y),
            BOX_COLOR,
            BOX_THICKNESS
        )

    def draw_center_line(self):
        cv2.line(
            self.img,
            (self.center_x, 0),
            (self.center_x, self.height),
            CENTER_LINE_COLOR,
            LINE_THICKNESS
        )

    def process(self):
        self.load_img()
        self.resize_img()
        self.get_img_shape()
        self.calculate_center()
        self.draw_center_region()
        self.draw_center_line()

        return self.img

    def display(self):
        cv2.imshow(self.img_path, self.img)


def main():
    image_paths = [
        "DATA/elevator.jpg",
        "DATA/chair.jpeg",
        "DATA/elevator_raw.jpg"
    ]

    processed_images = []

    for path in image_paths:
        try:
            processor = ImageProcessor(path)
            img = processor.process()
            processor.display()
            processed_images.append(img)

        except FileNotFoundError as error:
            print(error)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()