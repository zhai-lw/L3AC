__all__ = ['tree']


class MetaObject(type):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class Object(metaclass=MetaObject):
    def __post_init__(self):
        raise NotImplementedError


class BlackHole:
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        return self

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __str__(self):
        return ''

    def __repr__(self):
        return 'BlackHole()'
