class WindowedAdaptation():
    def __init__(self, warmup, windowsize = 25):
        self._windowsize = windowsize
        self._warmup = warmup
        self._closewindow = self._windowsize
        self._idx = 0
        self._closures = []
        self._num_windows = 0
        self._calculate_windows()

    def _calculate_windows(self):
        for w in range(self._warmup + 1):
            if w == self._closewindow:
                self._closures.append(w)
                self._calculate_next_window()
        self._num_windows = len(self._closures)

    def _calculate_next_window(self):
        self._windowsize *= 2
        nextclosewindow = self._closewindow + self._windowsize
        if self._closewindow + 2 * self._windowsize >= self._warmup:
            self._closewindow = self._warmup
        else:
            self._closewindow = nextclosewindow

    def window_closed(self, m):
        closed = m == self._closures[self._idx]
        if closed and self._idx < self._num_windows - 1:
            self._idx += 1
        return closed

    def reset(self):
        self._idx = 0
        self._closures.clear()
        self._num_windows = 0
        self._calculate_windows()

if __name__ == "__main__":
    warmup = 15_000
    iterations = warmup + 1
    wa = WindowedAdaptation(warmup)
    closures = wa._closures
    for m in range(iterations):
        if wa.window_closed(m):
            print(m)

    wa.reset()
    print(wa._closures == closures)
