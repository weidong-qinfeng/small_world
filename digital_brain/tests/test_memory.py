"""记忆系统单元测试 - DeclarativeMemory"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pytest

from digital_brain.src.core.memory.declarative_memory import DeclarativeMemory, WorldModel
from digital_brain.src.core.models import Entity, EntityType, Relation, RelationType


@pytest.fixture
def mem():
    return DeclarativeMemory()


def _make_entity(eid, name, etype=EntityType.ABSTRACT, **attrs):
    return Entity(id=eid, name=name, entity_type=etype, attributes=dict(attrs))


def test_add_and_get_entity(mem):
    e = _make_entity("num_1", "1", value=1)
    rid = mem.add_entity(e)
    assert rid == "num_1"
    assert mem.entity_count == 1
    got = mem.get_entity("num_1")
    assert got is not None and got.name == "1"
    assert got.attributes["value"] == 1


def test_find_entity_by_name_and_alias(mem):
    mem.add_entity(_make_entity("num_2", "2"))
    e2 = Entity(id="num_2_cn", name="二", aliases=["2", "两"], entity_type=EntityType.ABSTRACT)
    mem.add_entity(e2)
    by_name = mem.find_entity_by_name("二")
    assert any(e.id == "num_2_cn" for e in by_name)
    by_alias = mem.find_entity_by_name("两")
    assert any(e.id == "num_2_cn" for e in by_alias)
    by_alias_2 = mem.find_entity_by_name("2")
    assert len(by_alias_2) >= 1


def test_update_entity(mem):
    e = _make_entity("num_1", "1", value=1)
    mem.add_entity(e)
    e2 = Entity(**e.dict())
    e2.aliases = ["一"]
    mem.update_entity(e2)
    got = mem.get_entity("num_1")
    assert "一" in got.aliases
    # 别名索引生效
    assert mem.find_entity_by_name("一")


def test_add_relation_and_neighbors(mem):
    mem.add_entity(_make_entity("n1", "1"))
    mem.add_entity(_make_entity("n2", "2"))
    mem.add_entity(_make_entity("n3", "3"))
    mem.add_relation(Relation(id="r1", source_id="n1", target_id="n2",
                              relation_type=RelationType.SUCCESSOR))
    mem.add_relation(Relation(id="r2", source_id="n2", target_id="n3",
                              relation_type=RelationType.SUCCESSOR))
    rels = mem.find_relations_of("n2")
    assert len(rels) == 2  # 1入 (from n1) + 1出 (to n3)
    # 邻居
    nb = mem.get_neighbors("n1", hops=2)
    assert "n2" in nb.get(1, [])
    assert "n3" in nb.get(2, [])


def test_entity_type_filter(mem):
    mem.add_entity(_make_entity("n1", "1", EntityType.ABSTRACT))
    mem.add_entity(Entity(id="f1", name="finger_1", entity_type=EntityType.PHYSICAL))
    abs_ = mem.find_entities_by_type(EntityType.ABSTRACT)
    phys = mem.find_entities_by_type(EntityType.PHYSICAL)
    assert len(abs_) == 1
    assert len(phys) == 1


def test_crud_delete(mem):
    e = _make_entity("x", "x")
    mem.add_entity(e)
    assert mem.delete_entity("x")
    assert mem.entity_count == 0
    assert not mem.delete_entity("x")


def test_world_model_describe(mem):
    mem.add_entity(_make_entity("n1", "1"))
    mem.add_entity(_make_entity("n2", "2"))
    mem.add_relation(Relation(id="r1", source_id="n1", target_id="n2",
                              relation_type=RelationType.SUCCESSOR))
    wm = WorldModel(mem)
    desc = wm.describe_entity("n1")
    assert "1" in desc and "successor" in desc
