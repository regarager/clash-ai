import numpy as np


class BBox:
    """
    Represents a bounding box.
    (x1, y1) is the top-left corner.
    (x2, y2) is the bottom-right corner.
    """

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    @classmethod
    def from_xywh(cls, x1: int, y1: int, w: int, h: int):
        return cls(x1, y1, x1 + w, y1 + h)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_xywh(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.width, self.height

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def pt(self, x: float, y: float):
        return (
            int(self.x1 + x * self.width),
            int(self.y1 + y * self.height),
        )

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2


BATTLE = (710, 1345)
CARDS = [
    (525, 1550),
    (705, 1550),
    (900, 1550),
    (1085, 1550),
]

FIELD_CORNERS = BBox(300, 180, 1140, 1300)
ALLY_CORNERS = BBox(300, 770, 1140, 1300)

ELIXIR_BAR_BBOX = BBox(484, 1674, 1168, 1712)
MAIN_PAGE_BBOX = BBox.from_xywh(635, 1285, 100, 5)
END_SCREEN_BBOX = BBox.from_xywh(800, 1540, 100, 5)

TOWER_HEALTH_WIDTH = 104
TOWER_HEALTH_HEIGHT = 2  # purposefully less to avoid text

TOWER_BBOXES = {
    "ally_king_tower": BBox.from_xywh(663, 1309, 144, TOWER_HEALTH_HEIGHT),
    "ally_left_princess_tower": BBox.from_xywh(
        423, 1077, TOWER_HEALTH_WIDTH, TOWER_HEALTH_HEIGHT
    ),
    "ally_right_princess_tower": BBox.from_xywh(
        936, 1077, TOWER_HEALTH_WIDTH, TOWER_HEALTH_HEIGHT
    ),
    "enemy_king_tower": BBox.from_xywh(663, 65, 144, TOWER_HEALTH_HEIGHT),
    "enemy_left_princess_tower": BBox.from_xywh(
        423, 264, TOWER_HEALTH_WIDTH, TOWER_HEALTH_HEIGHT
    ),
    "enemy_right_princess_tower": BBox.from_xywh(
        936, 264, TOWER_HEALTH_WIDTH, TOWER_HEALTH_HEIGHT
    ),
}
