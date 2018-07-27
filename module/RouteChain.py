class RouteChain():
    
    def __init__(self, path=''):
        self._path = path

    def __getattr__(self, path):
        return RouteChain('%s/%s' % (self._path, path))

    def __str__(self):
        return self._path

    __repr__ = __str__


if __name__ == '__main__':
    print(RouteChain().fuck.you.man)
