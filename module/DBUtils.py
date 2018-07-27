
import os
import sys
import prestodb


class Properties():

    def __init__(self, config_file):
        self.__config_file = config_file
        self.__properties = {}

        with open(self.__config_file, 'r') as config_file:
            for line in config_file.readlines():
                line = line.strip().replace('\n', '')
                if line.find("#") != -1:
                    line = line[0:line.find('#')]
                if line.find('=') > 0:
                    strs = line.split('=')
                    self.__properties[strs[0].strip()] = strs[1].strip()

    @property
    def properties(self):
        return self.__properties

    @properties.setter
    def properties(self, params_dict):
        self.__properties = params_dict


class PrestoUtils(Properties):
    
    def __init__(self, config_file=None, host=None, port=None, user=None, catalog=None):

        if (config_file is not None) and (host or port or user or catalog is not None):
            raise Exception("Do not provide both of config file and other arguments!!!")

        elif config_file is not None:
            super(PrestoUtils, self).__init__(config_file)

        elif host and port and user and catalog is not None:
            params_dict = {
                'presto.host': host,
                'presto.port': port,
                'presto.user': user,
                'presto.catalog': catalog
            }
            self.properties = params_dict

        else:
            raise Exception("Please provide config file or arguments!!!")

        self.__presto_conn = prestodb.dbapi.connect(
            host=self.properties['presto.host'],
            port=self.properties['presto.port'],
            user=self.properties['presto.user'],
            catalog=self.properties['presto.catalog'] 
        )
            
    @property
    def presto_conn(self):
        return self.__presto_conn

    def create_table(self, sql):
        cur = self.__presto_conn.cursor()
        cur.execute(sql)
        print(cur.fetchall())

    def drop_table(self, sql):
        cur = self.__presto_conn.cursor()
        cur.execute(sql)
        print(cur.fetchall())


if __name__ == '__main__':
    PRESTO_CONFIG_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'presto_prod.config'))
    pu = PrestoUtils(config_file=PRESTO_CONFIG_FILE)
    pu2 = PrestoUtils(host='10.10.22.8', port='10300', user='prod', catalog='prod_hive')
    # pu3 = PrestoUtils(config_file=PRESTO_CONFIG_FILE, host='666')
    # pu4 = PrestoUtils()
    print(pu.properties)
    print(pu2.properties)

