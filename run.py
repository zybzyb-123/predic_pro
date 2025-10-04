import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import numpy as np

from lru import get_lru_topk_by_simulation
from arc import ARC
from lecar import LeCaR


def load_trace_dup(path):
    df = pd.read_csv(path)
    trace = df[['timestamp', 'pageId']].dropna().values.tolist()
    return remove_consecutive_duplicates(trace)

def load_trace(path):
    df = pd.read_csv(path)
    trace = df[['timestamp', 'pageId']].dropna().values.tolist()
    return trace

def remove_consecutive_duplicates(trace):
    result = [trace[0]]
    for i in range(1, len(trace)):
        if trace[i][1] != trace[i - 1][1]:
            result.append(trace[i])
    return result

class SimulatedCache:
    def __init__(self, size, prefetch_k=0):
        self.size = size
        self.prefetch_k = prefetch_k
        self.cache = set()
        self.queue = deque()
        self.hits = 0
        self.total = 0

    def preload(self, pages):
        for p in pages:
            self._insert(p)

    def access(self, page_id):
        self.total += 1
        if page_id in self.cache:
            self.hits += 1
        else:
            self._insert(page_id)
            self.prefetch(page_id)

    def _insert(self, page_id):
        if page_id in self.cache:
            return
        if len(self.cache) >= self.size:
            old = self.queue.popleft()
            self.cache.remove(old)
        self.cache.add(page_id)
        self.queue.append(page_id)
    
    def prefetch(self, current_pid):
        for i in range(1, self.prefetch_k + 1):
            next_pid = current_pid + i
            if next_pid not in self.cache:
                self._insert(next_pid)

    def stats(self):
        return self.hits / self.total

def simulate_cache(trace, preload_pages, cache_size):
    cache = SimulatedCache(cache_size)
    cache.preload(preload_pages)
    hit_curve = []
    first_timestamp = None
    for i, (time, pid) in enumerate(trace):
        if first_timestamp is None:
            first_timestamp = time
        cache.access(pid)
        if (i + 1) % 100 == 0:
            hit_curve.append(cache.hits / cache.total)
        if (time - first_timestamp) / 1800 > 1:
            break
    return hit_curve, cache.stats()

def simulate_seq_cache(trace, cache_size, prefetch_k=4):
    cache = SimulatedCache(cache_size,prefetch_k)
    hit_curve = []
    for i, (_, pid) in enumerate(trace):
        cache.access(pid)
        if (i + 1) % 100 == 0:
            hit_curve.append(cache.hits / cache.total)
    return hit_curve, cache.stats()

def simulate_cache_only(trace, preload_pages):
    preload_set = set(preload_pages)
    relevant_total = 0
    relevant_hits = 0
    hit_curve = []
    times_curve = []
    first_timestamp = None
    for i, (time, pid) in enumerate(trace):
        if first_timestamp is None:
                first_timestamp = time
        if pid in preload_set:
            relevant_hits += 1
        relevant_total += 1

        if (i + 1) % 100 == 0:
            times_curve.append((time - first_timestamp) / 60)  # 记录时间戳
            hit_curve.append(relevant_hits / relevant_total if relevant_total > 0 else 0)
        if (time - first_timestamp) / 1800 > 1:
            break
    return hit_curve, (relevant_hits / relevant_total if relevant_total > 0 else 0)

def plot_hit_rate_curves(curves, times_curve):
    plt.figure(figsize=(10, 5))
    for label, curve in curves.items():
        plt.plot(times_curve, [v * 100 for v in curve], label=label) 
    plt.title("Cache hit ratio comparison")
    plt.xlabel("Requests (per minute)")
    plt.ylabel("Hit ratio(%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_hit_rate_curves(curves):
    plt.figure(figsize=(10, 5))
    for label, curve in curves.items():
        plt.plot([v * 100 for v in curve], label=label) 
    plt.title("Cache hit ratio comparison")
    plt.xlabel("Requests (per 100)")
    plt.ylabel("Hit ratio(%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # 数据集1
    # warm_trace = load_trace("trace/trace13h.csv")
    # test_trace = load_trace("trace/trace14h.csv")

    # 数据集2
    # warm_trace = load_trace("trace/trace12h.csv")
    # test_trace = load_trace("trace/trace13h.csv")

    # 数据集3
    # warm_trace = load_trace("trace/trace16h.csv")
    # test_trace = load_trace("trace/trace17h.csv")

    # 数据集4
    # warm_trace = load_trace("trace/trace17h.csv")
    # test_trace = load_trace("trace/trace18h.csv")

    # 数据集5
    # warm_trace = load_trace("trace/trace18h.csv")
    # test_trace = load_trace("trace/trace19h.csv")

    # 数据集6 均偏低
    # warm_trace = load_trace("trace/trace19h.csv")
    # warm_trace = warm_trace[292367:]
    # test_trace = load_trace("trace/trace20h.csv")

    # 数据集7
    # warm_trace = load_trace("trace/trace21h.csv")
    # test_trace = load_trace("trace/trace22h.csv")

    # 数据集8
    # warm_trace = load_trace("trace/trace22h.csv")
    # warm_trace = warm_trace[92702:]
    # test_trace = load_trace("trace/trace23h.csv")

    # 数据集9 三种情况不变
    # warm_trace = load_trace("trace/trace6h.csv")
    # test_trace = load_trace("trace/trace7h.csv")

    # 数据集10
    # warm_trace = load_trace("trace/trace12h.csv")
    # test_trace = load_trace("trace/trace20h.csv")

    #数据集11
    warm_trace = load_trace("trace/similar/warm7.txt")
    test_trace = load_trace("trace/similar/test7.txt")
    
    # top_k = int(cache_size * 0.5)
    lru_hits = []
    arc_hits = []
    lecar_hits = []
    cache_size = 1024

    arc = ARC(cache_size = cache_size) 
    for _,pageId in warm_trace:
        arc.request(pageId)

    lecar = LeCaR(cache_size = cache_size) 
    for _,pageId in warm_trace:
        lecar.request(pageId)

    
    for a in [0.25, 0.5, 0.75, 1]:
        top_k = int(cache_size * a)
        print("使用 LRU 快照...")
        lru_prefetch = get_lru_topk_by_simulation(warm_trace, top_k)
        lru_curve, lru_hit = simulate_cache(test_trace, lru_prefetch, cache_size)
        # lru_curve, lru_hit = simulate_cache_only(test_trace, lru_prefetch)
        lru_hits.append(lru_hit)
        
        print("使用 ARC 快照...")
        arc_prefetch = arc.getHotPage(top_k)
        # arc2 = ARC(cache_size = cache_size) 
        # for pageId in arc_prefetch:
        #     arc2.request(pageId)
        
        # hits = 0
        # misses = 0
        # arc_hit = 0
        # for _,pageId in test_trace:
        #     miss, evicted = arc2.request(pageId)
        #     if miss:
        #         misses += 1
        #     else:
        #         hits += 1
        # arc_hit = hits / (hits + misses)

        # arc_curve, arc_hit = simulate_cache_only(test_trace, arc_prefetch)
        arc_curve, arc_hit = simulate_cache(test_trace, arc_prefetch, cache_size)
        arc_hits.append(arc_hit)

        print("使用 LeCaR 快照...")
        
        
        lecar_prefetch = lecar.getHotPages(top_k)
        # lecar2 = LeCaR(cache_size = cache_size) 
        # for pageId in lecar_prefetch:
        #     lecar2.request(pageId)
        
        # hits = 0
        # misses = 0
        # lecar_hit = 0
        # for _,pageId in test_trace:
        #     miss, evicted = lecar2.request(pageId)
        #     if miss:
        #         misses += 1
        #     else:
        #         hits += 1
        # lecar_hit = hits / (hits + misses)
        # lecar_curve, lecar_hit = simulate_cache_only(test_trace, lecar_prefetch)
        lecar_curve, lecar_hit = simulate_cache(test_trace, lecar_prefetch, cache_size)
        lecar_hits.append(lecar_hit)

        # print("命中率比较：")
        # print(f"LRU: {lru_hit:.4f}")
        # print(f"ARC: {arc_hit:.4f}")
        # print(f"LeCaR: {lecar_hit:.4f}")

        # plot_hit_rate_curves({
        #     "LRU": lru_curve,
        #     "ARC": arc_curve,
        #     "LeCaR": lecar_curve
        # })
    print("命中率比较：")
    print(f"LRU: {lru_hits}")
    print(f"ARC: {arc_hits}")
    print(f"LeCaR: {lecar_hits}")
