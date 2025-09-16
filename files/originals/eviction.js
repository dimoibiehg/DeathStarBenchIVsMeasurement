db.adminCommand({
    "setParameter": 1,
    "wiredTigerEngineRuntimeConfig": "eviction=(threads_min=6,threads_max=12),eviction_dirty_target=20,eviction_dirty_trigger=80"
})

db.adminCommand({
    setParameter: 1,
    wiredTigerCacheSizeGB: 1
})