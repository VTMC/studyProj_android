class MovingAverageFilter:
    def __init__(self, size):
        if not isinstance(size, int):
            raise ValueError('size must be an integer.')
        
        self._size = size
        self._buffer = []

    def push(self, data):
        if not isinstance(data, (int, float)):
            raise ValueError('data must be a number.')

        self._buffer.append(data)

        if self._size < len(self._buffer):
            # remove first element
            self._buffer.pop(0)

        return self.average()

    def average(self):
        count = 0
        total = 0

        for element in self._buffer:
            if isinstance(element, (int, float)) and not (element == float('nan') or element == float('inf')):
                count += 1
                total += element

        return total / count if count > 0 else 0