import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module import const

if __name__ == '__main__':
    const.pi = 3.14
    print(const.pi)
    const.pi = 3.15