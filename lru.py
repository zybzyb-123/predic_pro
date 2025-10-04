from collections import deque, Counter

class LRUCacheForSnapshot:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.queue = deque()
        self.access_counter = Counter()

    def access(self, page_id):
        self.access_counter[page_id] += 1
        if page_id in self.cache:
            self.queue.remove(page_id)
        elif len(self.queue) >= self.capacity:
            old = self.queue.popleft()
            del self.cache[old]
        self.queue.append(page_id)
        self.cache[page_id] = True

    def get_hot_pages(self, top_k):
        return [page for page, _ in self.access_counter.most_common(top_k)]

def get_lru_topk_by_simulation(trace, cache_size=1000, top_k=250):
    lru_sim = LRUCacheForSnapshot(capacity=cache_size)
    for _, page_id in trace:
        lru_sim.access(page_id)
    return lru_sim.get_hot_pages(top_k)