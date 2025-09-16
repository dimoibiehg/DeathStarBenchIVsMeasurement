db.adminCommand({
    "setParameter": 1,
    "wiredTigerEngineRuntimeConfig": "eviction_dirty_target={{ eviction_dirty_target }},eviction_dirty_trigger= {{ eviction_dirty_trigger }}"
})

db.adminCommand({
    setParameter: 1,
    wiredTigerCacheSizeGB: {{ wiredTigerCacheSizeGB }}
})