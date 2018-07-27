import sys

__author__ = 'tankeryang 2018-03-28'

class _const(object):
    class ConstError(PermissionError):
        pass
    
    def __setattr__(self, name, value):
        if name in self.__dict__.keys():
            raise self.ConstError("Can't rebind const(%s)" % name)
        self.__dict__[name] = value
    
    def __delattr__(self, name):
        if name in self.__dict__:
            raise self.ConstError("Can't unbind const(%s)" % name)
        raise NameError(name)

sys.modules[__name__]=_const()
