"""Contagem de IDs que cruzam uma linha horizontal."""


class LineCounter:
    """Conta cada ID uma única vez ao cruzar uma linha horizontal.

    line_y é uma coordenada vertical em pixels.
    direction='down' conta cruzamentos de cima para baixo;
    direction='up' conta cruzamentos de baixo para cima.
    """
    def __init__(self, line_y: int, direction: str = "down"):
        if direction not in {"down", "up"}:
            raise ValueError("direction deve ser 'down' ou 'up'.")
        self.line_y = int(line_y)
        self.direction = direction
        self.previous_y = {}
        self.counted_ids = set()

    def update(self, track_id: int, center_y: float) -> bool:
        """Atualiza a posição do ID; retorna True apenas no primeiro cruzamento."""
        track_id = int(track_id)
        center_y = float(center_y)
        previous = self.previous_y.get(track_id)
        self.previous_y[track_id] = center_y

        if previous is None or track_id in self.counted_ids:
            return False

        crossed_down = previous < self.line_y <= center_y
        crossed_up = previous > self.line_y >= center_y
        crossed = crossed_down if self.direction == "down" else crossed_up

        if crossed:
            self.counted_ids.add(track_id)
            return True
        return False
