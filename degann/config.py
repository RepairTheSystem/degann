from degann.networks.losses import LossFactory
from degann.networks.optimizers import OptimizerFactory
from degann.networks.activations import ActivationFactory
from degann.networks.layers.tf_dense import DenseFactory
#  from degann.networks.metrics import MetricFactory


class ConfigManager:
    def __init__(self, framework: str):
        # Активируем фабрики для выбранного фреймворка
        self.loss_factory = LossFactory(framework=framework)
        self.optimizer_factory = OptimizerFactory(framework=framework)
        self.activation_factory = ActivationFactory(framework=framework)
        self.dense_factory = DenseFactory(framework=framework)
        #  self.metric_factory = MetricFactory(framework=framework)

        # Инициализация методов фабрик для упрощенного доступа
        self.loss = self.loss_factory.get_all_loss_functions()
        self.optimizer = self.optimizer_factory.get_all_optimizers()
        self.activation = self.activation_factory.get_all_activation_functions
        #  self.metric = self.metric_factory.get_all_metrics()

        def create_dense_layer(self, **kwargs):
            return self.dense_factory.create_dense_layer(**kwargs)

# Инициализация для пользователя
def load_config(framework: str) -> ConfigManager:
    return ConfigManager(framework)

# Пример вызова: config = load_config('tensorflow')
