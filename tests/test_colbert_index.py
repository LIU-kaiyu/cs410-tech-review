from src.colbert.index import ColBERTIndexConfig, ColBERTIndexer
from src.data.schema import Doc


class _FakeConfig:
    nbits = 2


class _FakeCheckpoint:
    colbert_config = _FakeConfig()


class _FakeWrappedColBERT:
    def __init__(self):
        self.config = _FakeConfig()
        self.inference_ckpt = _FakeCheckpoint()


class _FakeRAGModel:
    def __init__(self):
        self.model = _FakeWrappedColBERT()
        self.index_called_with_nbits = None

    def index(self, **kwargs):
        self.index_called_with_nbits = self.model.config.nbits
        return ".ragatouille/colbert/indexes/fake"


class _FakeColBERTIndexer(ColBERTIndexer):
    def __init__(self, config):
        super().__init__(config=config)
        self.fake_model = _FakeRAGModel()

    def _load_model(self):
        self._model = self.fake_model
        return self.fake_model


def test_build_index_sets_effective_nbits_before_ragatouille_index_call():
    indexer = _FakeColBERTIndexer(
        ColBERTIndexConfig(index_name="fake", nbits=4, index_root=".tmp-test-index")
    )
    docs = [Doc(doc_id="d1", title="Title", text="Body text")]

    stats = indexer.build_index(docs)

    assert indexer.fake_model.index_called_with_nbits == 4
    assert stats.nbits == 4
    assert stats.effective_nbits == 4
