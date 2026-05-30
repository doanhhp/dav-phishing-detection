"""Feature processor factory for modular feature engineering."""

class FeatureFactory:
    """Factory for instantiating feature processors based on model type."""

    @staticmethod
    def get_processor(processor_name: str, config: dict):
        """
        Get a feature processor instance.

        Args:
            processor_name: Type of processor ('manual', 'sequential', 'rnn_gru', 'multimodal', 'tfidf_svd')
            config: Configuration dictionary

        Returns:
            Feature processor instance
        """
        from .manual import ManualFeatureProcessor
        from .sequential import SequentialTokenProcessor
        from .rnn_gru import RnnGruProcessor
        from .multimodal import MultimodalProcessor
        from .tfidf_svd import TfidfSvdProcessor
        from .tfidf_vae import TfidfVaeProcessor

        processors = {
            "manual": ManualFeatureProcessor,
            "sequential": SequentialTokenProcessor,
            "rnn_gru": RnnGruProcessor,
            "multimodal": MultimodalProcessor,
            "tfidf_svd": TfidfSvdProcessor,
            "tfidf_vae": TfidfVaeProcessor,
        }

        if processor_name not in processors:
            raise ValueError(f"Unknown processor: {processor_name}. Available: {list(processors.keys())}")

        return processors[processor_name](config)
