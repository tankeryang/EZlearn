from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import scale, robust_scale, minmax_scale

PROCESSOR = {
    'MMS': MinMaxScaler(feature_range=(0, 1)),
    'STDS': StandardScaler(),
    'NML': Normalizer(),
    'RS': RobustScaler(),
    'LB': LabelBinarizer(),
    'LE': LabelEncoder()
}