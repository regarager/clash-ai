class BBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int) -> None:
        # (0, 0) is top left in emulator
        self.x1: int = min(x1, x2)
        self.y1: int = max(y1, y2)
        self.x2: int = max(x1, x2)
        self.y2: int = min(x1, x2)

    def bl(self):
        return (self.x1, self.y1)

    def br(self):
        return (self.x2, self.y1)

    def tl(self):
        return (self.x1, self.y2)

    def tr(self):
        return (self.x2, self.y2)

    def pt(self, x: float, y: float):
        return (
            int(self.x1 + x * (self.x2 - self.x1)),
            int(self.y1 + y * (self.y2 - self.y1)),
        )


BATTLE = (700, 1350)
CARDS = [
    (530, 1530),
    (700, 1530),
    (900, 1530),
    (1100, 1530),
]

FIELD_CORNERS = BBox(300, 180, 1140, 1300)
ALLY_CORNERS = BBox(300, 770, 1140, 1300)
