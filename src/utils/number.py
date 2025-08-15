class _MAX:
    def __lt__(self, other): return False

    def __le__(self, other): return False

    def __gt__(self, other): return True

    def __ge__(self, other): return True


MAX = _MAX()


class _MIN:
    def __lt__(self, other): return True

    def __le__(self, other): return True

    def __gt__(self, other): return False

    def __ge__(self, other): return False


MIN = _MIN()
