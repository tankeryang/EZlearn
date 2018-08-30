import os
import sys
# customer module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module import TrainerBase


class DecoratorTest(TrainerBase):

    def __init__(self):
        super(DecoratorTest, self).__init__()


    def logging(message):
        def decorator(func):
            def _func(self, *args, **kw):
                self.logger.info('='*60)
                self.logger.info(message)
                self.logger.info('='*60)
                func(self)
                self.logger.info(message + 'complete!')
            return _func
        return decorator


    @logging("test")
    def logging_test(self):
        print(self.args_dict)


if __name__ == '__main__':
    dt = DecoratorTest()
    dt.logging_test()