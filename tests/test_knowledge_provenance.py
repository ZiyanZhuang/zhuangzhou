import json
from unittest.mock import Mock, patch
import threading

from zhuangzhou.memory.kg_engine import LocalKGEngine
from zhuangzhou.memory.project_memory import ProjectMemory


def test_distilled_triples_stay_unverified_through_query_and_export(tmp_path, monkeypatch):
    import zhuangzhou.memory.kg_engine as module
    monkeypatch.setattr(module, 'LOCAL_GRAPH_PATH', str(tmp_path / 'graph.jsonl'))
    monkeypatch.setattr(module, 'AUTO_EXPORT_THRESHOLD', 0)
    engine = LocalKGEngine.__new__(LocalKGEngine)  # Exercise storage without starting a model worker.
    engine.model = 'explicit-test-stub'
    engine._save_triples([{'head': 'fixture', 'relation': 'suggests', 'tail': 'hypothesis'}], '2026-10-15')
    record = json.loads((tmp_path / 'graph.jsonl').read_text(encoding='utf-8'))
    assert record['verification_status'] == 'unverified'
    for triple in [*LocalKGEngine.query('fixture'), *LocalKGEngine.query_triples('fixture')]:
        assert triple['verification_status'] == 'unverified'
        assert triple['source_type'] == 'llm_distillation'
    exported = LocalKGEngine.export_overlay_for_graphrag(str(tmp_path / 'export'))
    export = json.loads(open(exported['export_jsonl'], encoding='utf-8').read())
    assert export['verification_status'] == 'unverified'
    memory = ProjectMemory(memory_dir=str(tmp_path / 'project'))
    assert 'unverified' in memory._get_local_graph_patterns()[0]


def test_kg_queue_overload_never_blocks_agent():
    import queue
    engine = LocalKGEngine.__new__(LocalKGEngine)
    engine.queue = Mock()
    engine.queue.put.side_effect = AssertionError('Must not block on queue.put')
    engine.queue.put_nowait.side_effect = queue.Full
    assert engine.add_to_queue({'input': 'fixture'}) is False


def test_query_access_counter_cannot_erase_concurrent_new_evidence(tmp_path, monkeypatch):
    import zhuangzhou.memory.kg_engine as module
    monkeypatch.setattr(module, 'LOCAL_GRAPH_PATH', str(tmp_path / 'graph.jsonl'))
    monkeypatch.setattr(module, 'AUTO_EXPORT_THRESHOLD', 0)
    engine = LocalKGEngine.__new__(LocalKGEngine)
    engine.model = 'test-stub'
    engine._save_triples([{'head': 'first', 'relation': 'is', 'tail': 'fixture'}], '2026-10-15')
    before_write, allow_write, appended = threading.Event(), threading.Event(), threading.Event()
    original_write = LocalKGEngine._write_records
    def blocked_write(records):
        before_write.set()
        assert allow_write.wait(2)
        original_write(records)
    def append():
        engine._save_triples([{'head': 'second', 'relation': 'is', 'tail': 'fixture'}], '2026-10-15')
        appended.set()
    with patch.object(LocalKGEngine, '_write_records', side_effect=blocked_write):
        query_thread = threading.Thread(target=lambda: LocalKGEngine.query('first'), daemon=True)
        query_thread.start()
        assert before_write.wait(2)
        append_thread = threading.Thread(target=append, daemon=True)
        append_thread.start()
        appended.wait(.1)
        allow_write.set()
        query_thread.join(2)
        append_thread.join(2)
        assert not query_thread.is_alive() and not append_thread.is_alive()
    assert {triple['head'] for triple in LocalKGEngine.iter_triples()} == {'first', 'second'}
