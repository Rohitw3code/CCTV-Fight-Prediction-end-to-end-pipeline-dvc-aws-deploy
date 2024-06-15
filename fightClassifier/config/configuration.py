from fightClassifier.utils.read_yaml import read_yaml

class ConfigurationManager:
    def __init__(self,CONFIG_PATH='config/config.yaml',PARAMS_PATH='params.yaml'):
        self.config = read_yaml(CONFIG_PATH)
        self.params = read_yaml(PARAMS_PATH)
    
    def config_data_ingestion(self)->str:
        config = self.config['data_ingestion']
        print(config)
        return config
