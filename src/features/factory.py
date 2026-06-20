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
        from .sequential import SequentialTokenProcessor
        from .rnn_gru import RnnGruProcessor
        from .multimodal import MultimodalProcessor
        from .structural import StructuralProcessor
        from .url_processor import UrlProcessor

        processors = {
            "sequential": SequentialTokenProcessor,
            "rnn_gru": RnnGruProcessor,
            "multimodal": MultimodalProcessor,
            "structural": StructuralProcessor,
            "url": UrlProcessor,
        }

        if processor_name not in processors:
            raise ValueError(f"Unknown processor: {processor_name}. Available: {list(processors.keys())}")

        return processors[processor_name](config)
